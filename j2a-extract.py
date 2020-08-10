from __future__ import print_function
import os
import sys
import itertools
import logging
from logging import error, warning, info
import argparse
import json
import wave
import j2a
from j2a import J2A, FrameConverter
import j2metadata

if sys.version_info[0] <= 2:
    input = raw_input

def mkdir(*pargs):
    dirname = os.path.join(*(str(arg) for arg in pargs))
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    return dirname

def get_default_foldername(dest_folder, filename, default_foldername, func):
    if dest_folder is not None:
        foldername = dest_folder
    elif isinstance(filename, int):
        foldername = default_foldername
    else:
        foldername = func(filename)

    foldername = foldername.rstrip(os.path.sep)
    return foldername
    # for retval in itertools.chain((foldername,), ("%s_%d" % (foldername, i) for i in itertools.count(1))):
    #     if not os.path.lexists(retval):
    #         return retval

def legacy_extractor(animsfilename, outputdir = None, palette_file = "Diamondus_2.pal"):
    outputdir = get_default_foldername(outputdir, animsfilename, "Anims-j2a",
            lambda fname : os.path.join(os.path.dirname(fname), os.path.basename(fname).replace(".", "-"))
            )
    j2a = J2A(animsfilename).read()
    info("Extracting to: %s", outputdir)
    renderer = FrameConverter(palette_file = palette_file)
    for setnum, s in enumerate(j2a.sets):
        s = j2a.sets[setnum]
        setdir = os.path.join(outputdir, str(setnum))
        if not os.path.exists(setdir):
            os.makedirs(setdir)
        for animnum, anim in enumerate(s.animations):
            animdir = os.path.join(setdir, str(animnum))
            if not os.path.exists(animdir):
                os.makedirs(animdir)
            fps_filename = os.path.join(animdir, "fps.%d" % anim.fps)
            open(fps_filename, "a").close() # Touch fps file, leave it empty
            for framenum, frame in enumerate(anim.frames):
                frameid = str(framenum)
                if frame.tagged:
                    frameid += "t"
                imgfilename = os.path.join(animdir, "{0:s},{1:d},{2:d},{3:d},{4:d},{5:d},{6:d}.png".format(
                    frameid,
                    *frame.origin +
                     frame.coldspot +
                     frame.gunspot
                ))
                renderer.to_image(frame).save(imgfilename)
        info("Finished extracting set %d (%d animations)", setnum, animnum + 1)
    return 0

def metadata_based_extractor(animsfilename, dest_folder, palette_file, metadata_writer, path_formats, extra_args):
    settings_separate, no_clobber = (getattr(extra_args, k) for k in ("settings", "no_clobber"))
    dest_folder = get_default_foldername(dest_folder, animsfilename, "Anims",
            lambda fname : fname[:-4] if fname.lower().endswith(".j2a") else fname + "_dir"
            )
    anims = J2A(animsfilename).read()
    fconv = FrameConverter(palette_file = palette_file)
    info("Extracting to: %s", dest_folder)

    # Pregenerate metadata
    info("Generating metadata")
    j2a_metadata_interface = j2metadata.MetadataInterface( "J2A",
            children = j2metadata.j2a_metadata_interface.children,
            md_generator = lambda _ : { "separate_settings": list(settings_separate) }
            )
    anims_metadata = j2a_metadata_interface.generate_metadata(anims, dict(j2metadata.globalDefaults))

    def frame_writer(frame, path):
        # Setting transparency explicitly to work around limitations of old versions of Pillow
        fconv.to_image(frame).save(path, transparency = 0)

    def sample_writer(sample, path):
        assert sample._bits == 8, "Only 8-bit signed PCM supported"
        wave_out = wave.open(path, "wb")
        try:
            wave_out.setparams( (
                    sample._channels,
                    sample._bits >> 3,
                    sample._rate,
                    len(sample._data) // (sample._channels * (sample._bits >> 3)),
                    "NONE",
                    "not compressed"
                ) )
            wave_out.writeframesraw(bytearray(b ^ 0x80 for b in bytearray(sample._data)))
        finally:
            wave_out.close()

    if os.path.sep != '/':
        path_formats = dict((k, v.replace("/", os.path.sep)) for k,v in path_formats.items())

    tasks = []
    settings_fmt = "%ssettings." + metadata_writer.extension

    def coiterate_sub_md(obj_sub, base_metadata, base_path, type):
        # Iterate through the object list and the corresponding metadata subs at the same time
        sub_md = base_metadata[type + "s"]
        fmt = path_formats[type]
        if type in settings_separate:
            for child_num, child_obj in enumerate(obj_sub):
                child_md = sub_md.pop(child_num, {})
                child_path = fmt % (base_path, child_num)
                if child_md:
                    tasks.append((metadata_writer, child_md, settings_fmt % child_path))
                yield (child_obj, child_md, child_path)
        else:
            for child_num, child_obj in enumerate(obj_sub):
                child_md = sub_md.get(child_num, {})
                child_path = fmt % (base_path, child_num)
                yield (child_obj, child_md, child_path)

    base_path = dest_folder.rstrip(os.path.sep) + os.path.sep
    for s,set_md,set_path in coiterate_sub_md(anims.sets, anims_metadata, base_path, "set"):
        for anim,anim_md,anim_path in coiterate_sub_md(s.animations, set_md, set_path, "animation"):
            for frame,frame_md,frame_path in coiterate_sub_md(anim.frames, anim_md, anim_path, "frame"):
                tasks.append((frame_writer, frame, frame_path))
        for sample,sample_md,sample_path in coiterate_sub_md(s.samples, set_md, set_path, "sample"):
            tasks.append((sample_writer, sample, sample_path))

    tasks.append((metadata_writer, anims_metadata, settings_fmt % base_path))

    # Check for existing files
    if no_clobber:
        info("Checking for pre-existing files")
        for method, data, path in tasks:
            if os.path.isfile(path):
                raise FileExistsError(path)

    # Create all directories beforehand
    info("Populating directories")
    existing_dirs = dict()
    for method, data, path in tasks:
        pos = len(base_path) - 2
        while True:
            pos = path.find(os.path.sep, pos + 1)
            if pos == -1:
                break
            dirname = path[:pos]
            dirstatus = existing_dirs.get(dirname)

            if dirstatus is None:  # Not known yet
                dirstatus = not os.path.isdir(dirname)
#                 info("Checking directory '%s' for existence: %s", dirname, "no" if dirstatus else "yes")
                if dirstatus:
#                     info("Creating directory '%s'", dirname)
                    os.mkdir(dirname)
                existing_dirs[dirname] = dirstatus

    del existing_dirs

    # Invoke all scheduled tasks
    info("Exporting data and metadata")
    for method, data, path in tasks:
        method(data, path)
#         info("Invoking %s -> %s", method.__name__, path)

    return 0

def json_extractor(*pargs, **kwargs):
    def write_json_metadata_file(metadata, path):
        with open(path, "w") as f:
            json.dump(metadata, f, indent = 2)
            f.write('\n')
    write_json_metadata_file.extension = "json"

    return metadata_based_extractor(*pargs, metadata_writer=write_json_metadata_file, **kwargs)

def yaml_extractor(*pargs, **kwargs):
    import yaml

    try:  # Use LibYAML bindings if available (about 4.5X faster)
        from yaml import CDumper as Dumper
    except:
        Dumper = yaml.Dumper

    sequence_representer = lambda self, data : self.represent_sequence("tag:yaml.org,2002:seq", data, flow_style = True)
    Dumper.add_representer(tuple, sequence_representer)
    Dumper.add_representer(list, sequence_representer)

    def write_yaml_metadata_file(metadata, path, __Dumper = Dumper):
        with open(path, "w") as f:
            yaml.dump(metadata, f, Dumper = __Dumper, default_flow_style = False)
    write_yaml_metadata_file.extension = "yaml"

    return metadata_based_extractor(*pargs, metadata_writer=write_yaml_metadata_file, **kwargs)

def main(argv):
    # TODO: remove default palette
    # TODO: handle absolute paths?
    components = ("set", "animation", "frame", "sample")

    parser = argparse.ArgumentParser(add_help=False)
    parser.set_defaults(
        palette = os.path.join(os.path.dirname(sys.argv[0]), "Diamondus_2.pal"),
        log_level = logging.WARNING,
        exporter = json_extractor,
        settings = frozenset(("animation",)),
        no_clobber = True
    )

    class AdvOptAction(argparse._StoreAction):
        _used_opts = set()

        def __call__(self, parser, namespace, values, option_string=None):
            type(self)._used_opts.add(option_string)
            super(AdvOptAction, self).__call__(parser, namespace, values, option_string)

    class AdvOptConstAction(argparse._StoreConstAction):
        _used_opts = AdvOptAction._used_opts

        def __call__(self, parser, namespace, values, option_string=None):
            type(self)._used_opts.add(option_string)
            super(AdvOptConstAction, self).__call__(parser, namespace, values, option_string)

    main_args = parser.add_argument_group("Required arguments")
    main_args.add_argument("anims_file", nargs="?", help="path to the .j2a file to extract")
    main_args.add_argument("-o", "--output", "--folder", dest="output_folder", help="where to extract data (parent folder must exist)")
    main_args.add_argument("-p", "--palette-file", "--palette", dest="palette", help="palette file to use for extraction")

    opt_args = parser.add_argument_group("Optional arguments")
    opt_args.add_argument("-h", "--help", action="help", help="show this help message and exit")

    exporter_args = opt_args.add_mutually_exclusive_group() #("Exporter")
    exporter_args.add_argument("--legacy", action="store_const", const=legacy_extractor, dest="exporter", help="use legacy exporter")
    exporter_args.add_argument("--json",   action="store_const", const=json_extractor,   dest="exporter", help="use JSON-based exporter (default)")
    exporter_args.add_argument("--yaml",   action="store_const", const=yaml_extractor,   dest="exporter", help="use YAML-based exporter")

    opt_args.add_argument("--verbose", action="store_const", const=logging.INFO, dest="log_level",
        help="produce verbose console output (overrides previous --quiet)")
    opt_args.add_argument("--quiet", action="store_const", const=logging.ERROR, dest="log_level",
        help="suppress warning messages (overrides previous --verbose)")

    adv_group = parser.add_argument_group("Advanced settings", "Only available with the JSON and YAML exporter")

    def format_opt(fmt):
        try:
            fmt % 0
        except:
            raise argparse.ArgumentTypeError("invalid format: '%s'" % fmt)
        return fmt

    for elt in components:
        help_string = "override format used to generate each %s's path; default: '%s'" \
                % (elt, j2metadata.globalDefaults["%s_defaults" % elt]["path_template"].replace("%", "%%"))
        adv_group.add_argument(
                "--%s-path-format" % elt,
                dest="%s_path_fmt" % elt,
                type=format_opt,
                action=AdvOptAction,
                help=help_string,  # Avoid undesired substitution
                metavar="FORMAT"
                )

    def settings_opt(s):
        if s:
            strip_plural = lambda s : s[:-1] if s[-1:] == "s" else s
            l = frozenset(strip_plural(elt) for elt in s.split(","))
            if not l <= frozenset(components):
                raise argparse.ArgumentTypeError(
                        "Invalid argument '%s' specified, expected comma-separated list of options from the following: %s"
                        % (s, ", ".join(components))
                        )
            return l
        else:
            return frozenset()

    adv_group.add_argument("--settings-separate",
            type=settings_opt,
            action=AdvOptAction,
            metavar="LIST",
            dest="settings",
            help="comma-separated list specifying which components to generate separate config files for; default: '%s'" \
                    % ", ".join(parser.get_default("settings"))
            )

    adv_group.add_argument("--clobber", action=AdvOptConstAction, const=False, dest="no_clobber",
        help="overwrite existing files (default: no, unless using legacy exporter)")

    args = parser.parse_args(argv)

    if args.exporter is legacy_extractor and AdvOptAction._used_opts:
        parser.error(
                "the following options are not available for the legacy extractor: %s"
                % ", ".join(AdvOptAction._used_opts)
                )

    logging.basicConfig(format="%(levelname)s:j2a-export: %(message)s", level=args.log_level)
    j2a.logger.setLevel(args.log_level)

    if args.anims_file is None:
        animsfilename = input("Please type the path to the .j2a file you wish to extract (current folder: %s):\n" % os.getcwd())
    elif args.anims_file == "-":
        animsfilename = sys.stdin.buffer
    else:
        animsfilename = args.anims_file

#     from pprint import pprint
#     pprint(vars(args))
#     return 0

    if args.exporter is legacy_extractor:
        return legacy_extractor(animsfilename, args.output_folder, args.palette)
    else:
        path_formats = {}
        for elt in components:
            fmt = getattr(args, "%s_path_fmt" % elt)
            if fmt is None:
                path_formats[elt] = "%s" + j2metadata.globalDefaults["%s_defaults" % elt]["path_template"]
            else:
                path_formats[elt] = "%s" + fmt

        return args.exporter(animsfilename, args.output_folder, args.palette,
                path_formats=path_formats, extra_args=args)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
