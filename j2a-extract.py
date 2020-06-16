from __future__ import print_function
import os
import sys
# import itertools
import logging
from logging import error, warning, info
import argparse
import json
import wave
import j2a
from j2a import J2A, FrameConverter

if sys.version_info[0] <= 2:
    input = raw_input

def mkdir(*pargs):
    dirname = os.path.join(*(str(arg) for arg in pargs))
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    return dirname

def get_default_foldername(outputdir, filename, default_foldername, func):
    if outputdir is not None:
        foldername = outputdir
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

# Paths must use "/" as a separator; the code will replace with the current platform separator, if necessary
globalDefaults = dict(
    set_defaults = {
        "path_template": "set-%03d/"
    },
    animation_defaults = {
        "path_template": "animation-%03d/",
        "fps"          : 10
    },
    frame_defaults = {
        "path_template": "frame-%03d.png",
        "hotspot"      : (0, 0),
        "coldspot"     : (0, 0),
        "gunspot"      : (0, 0),
        "tagged"       : False,
        "mask"         : True
    },
    sample_defaults = {
        "path_template": "sample-%03d.wav",
        "volume"       : 32767,
        "loop_start"   : 0,
        "loop_end"     : None,
        "loop_bidi"    : False
    }
)

defaultKeys = {
    "set_defaults"      : tuple(k for k in globalDefaults["set_defaults"].keys()       if k != "path_template"),
    "animation_defaults": tuple(k for k in globalDefaults["animation_defaults"].keys() if k != "path_template"),
    "frame_defaults"    : tuple(k for k in globalDefaults["frame_defaults"].keys()     if k != "path_template"),
    "sample_defaults"   : tuple(k for k in globalDefaults["sample_defaults"].keys()    if k != "path_template")
}

class PropertiesCounter(object):
    __slots__ = ["_properties", "_threshold", "_min_count"]

    def __init__(self, properties, threshold, min_count):
        '''
        properties: an iterable returning the keys to look for
        threshold:  when querying modes, ignore for each key those that don't surpass
                    this quota over the total number of values given for the key
                    default: 0.5 (>50%)
        '''
        self._properties = dict((prop, dict()) for prop in properties)
        self._threshold = threshold
        self._min_count = min_count

    def update(self, d):
        for k,v in d.items():
            prop = self._properties[k]
            prop[v] = prop.get(v, 0) + 1

    def update_single(self, key, value):
        prop = self._properties[key]
        prop[value] = prop.get(value, 0) + 1

    def _get_modes_iter(self):
        for prop, values in self._properties.items():
            if values:
                total = sum(values.values())
                best_val, num_occurrences = max(values.items(), key = lambda t : t[1])
                if num_occurrences > self._min_count and num_occurrences > self._threshold * total:
                    yield (prop, best_val)

    def get_modes(self):
        return dict(self._get_modes_iter())

def strip_matching_defaults(d, defaults):
    '''
    d, defaults:    dict

    Return a dict of all key-value pairs in d that don't match what's already in defaults
    '''
    dummy = object()
    return dict((k, v) for k,v in d.items() if isinstance(v, dict) or defaults.get(k, dummy) != v)

def strip_matching_sub_defaults(md, defaults_iter):
    '''
    md:             metadata dict
    defaults_iter:  iterable of (str, dict) tuples

    For each (kind, defaults) tuple in defaults_iter, strip the matching defaults from md[kind].
    If this would leave it empty, remove it entirely.
    '''
    for sub_kind, sub_defaults in defaults_iter:
        new_sub_md = strip_matching_defaults(md[sub_kind], sub_defaults)
        if new_sub_md:
            md[sub_kind] = new_sub_md
        else:
            del md[sub_kind]

def condense_metadata(elements_md, main_kind, sub_kinds = (), threshold = 0.5, min_count = 1):
    '''
    Pop common key-value pairs among the elements of the given list, including
    nested defaults, and return a dict mapping each kind to its defaults.

    elements_md:    list of dicts, one for each element, describing its metadata;
                    will be modified during execution, to remove redundant defaults
    main_kind:      str, the name under which to wrap the defaults resulting from elements
    sub_kinds:      0 or more str, nested defaults to look for in each element (optional)
    '''
    # Initialize counters for each type of defaults
    main_defaults_counter = PropertiesCounter(defaultKeys[main_kind], threshold, min_count)
    sub_defaults_counters = dict( (
        kind,
        PropertiesCounter(defaultKeys[kind], threshold, min_count)
        ) for kind in sub_kinds)

    # Count the number of occurrences for each property
    for elt_md in elements_md:
        for k,v in elt_md.items():
            if k in sub_kinds:
                sub_defaults_counters[k].update(v)
            elif not isinstance(v, dict):
                main_defaults_counter.update_single(k, v)

    # Collect for each property the most common cases
    sub_metadata = dict((kind, counter.get_modes()) for kind,counter in sub_defaults_counters.items())
    main_metadata = main_defaults_counter.get_modes()

    # Keep only differences w.r.t. the global defaults
    elements_md[:] = [strip_matching_defaults(md, main_metadata) for md in elements_md]
    for elt_md in elements_md:
        strip_matching_sub_defaults(elt_md, sub_metadata.items())

    sub_metadata[main_kind] = main_metadata
    return sub_metadata

def iterable_to_dict_items(seq, base_idx = None):
    if base_idx is not None:
        yield ("base_idx", base_idx)
    yield ("count", len(seq))
    for i,v in enumerate(seq):
        if v:
            yield (i, v)

def generate_frame_metadata(frame):
    return {
        "hotspot" : frame.origin,
        "coldspot": frame.coldspot,
        "gunspot" : frame.gunspot,
        "tagged"  : frame.tagged,
        "mask"    : frame.mask is not None and any(frame.mask)  # FIXME: export mask if it's not trivial
    }

def generate_animation_metadata(anim):
    frames_md = [generate_frame_metadata(frame) for frame in anim.frames]
    metadata = condense_metadata(frames_md, "frame_defaults")
    metadata["frames"] = dict(iterable_to_dict_items(frames_md))
    metadata["fps"] = anim.fps
    return metadata

def generate_sample_metadata(sample):
    loop_start, loop_end, loop_bidi = sample.loop or (0, None, False)
    return {
        "volume"    : sample.volume,
        "loop_start": loop_start,
        "loop_end"  : loop_end,
        "loop_bidi" : loop_bidi
    }

def generate_set_metadata(s):
    animations_md = [generate_animation_metadata(anim) for anim   in s.animations]
    samples_md    = [generate_sample_metadata(sample)  for sample in s.samples   ]
    metadata = condense_metadata(animations_md, "animation_defaults", ("frame_defaults",))
    metadata.update(condense_metadata(samples_md, "sample_defaults", min_count = 0))
    metadata["animations"] = dict(iterable_to_dict_items(animations_md))
    metadata["samples"]    = dict(iterable_to_dict_items(samples_md, base_idx = s.samplesbaseindex))
    return metadata

def generate_metadata(anims):
    sets_md = [generate_set_metadata(s) for s in anims.sets]
    metadata = condense_metadata(sets_md, "set_defaults", ("animation_defaults", "frame_defaults", "sample_defaults"))
    metadata["sets"] = dict(iterable_to_dict_items(sets_md))

    strip_matching_sub_defaults(metadata, globalDefaults.items())

    # Special case for sets with exactly one sample
    # Otherwise, we have sample_defaults for just one sample
    for k,set_md in metadata["sets"].items():
        if k != "count" and set_md["samples"]["count"] == 1:
            sample = set_md["samples"].get(0, {})
            sample_defaults = set_md.pop("sample_defaults", {})
            if sample or sample_defaults:
                sample.update(sample_defaults)
                set_md["samples"][0] = sample

    return metadata

def metadata_based_extractor(animsfilename, outputdir, palette_file, metadata_writer, path_formats, settings_separate):
    outputdir = get_default_foldername(outputdir, animsfilename, "Anims",
            lambda fname : fname[:-4] if fname.lower().endswith(".j2a") else fname + "_dir"
            )
    anims = J2A(animsfilename).read()
    fconv = FrameConverter(palette_file = palette_file)
    info("Extracting to: %s", outputdir)

    # Pregenerate metadata
    info("Generating metadata")
    metadata = generate_metadata(anims)

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

    def get_sub_md_and_path(base_metadata, base_path, type, num):
        child_path = base_path + path_formats[type] % num
        try:
            if type in settings_separate:
                child_md = base_metadata[type + "s"].pop(str(num))
                tasks.append((metadata_writer, child_md, child_path))
            else:
                child_md = base_metadata[type + "s"][str(num)]
        except KeyError:
            child_md = {}

        return (child_md, child_path)

    base_path = outputdir.rstrip(os.path.sep) + os.path.sep
    for set_num,s in enumerate(anims.sets):
        set_md, set_path = get_sub_md_and_path(metadata, base_path, "set", set_num)

        for anim_num,anim in enumerate(s.animations):
            anim_md, anim_path = get_sub_md_and_path(set_md, set_path, "animation", anim_num)

            for frame_num,frame in enumerate(anim.frames):
                frame_md, frame_path = get_sub_md_and_path(anim_md, anim_path, "frame", frame_num)
                tasks.append((frame_writer, frame, frame_path))

        for sample_num,sample in enumerate(s.samples):
            sample_md, sample_path = get_sub_md_and_path(set_md, set_path, "sample", sample_num)
            tasks.append((sample_writer, sample, sample_path))

    tasks.append((metadata_writer, metadata, base_path))

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
        with open(path + "settings.json", "w") as f:
            json.dump(metadata, f, indent = 2)
            f.write('\n')

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
        with open(path + "settings.yaml", "w") as f:
            yaml.dump(metadata, f, Dumper = __Dumper, default_flow_style = False)

    return metadata_based_extractor(*pargs, metadata_writer=write_yaml_metadata_file, **kwargs)

def main(argv):
    # TODO: remove default palette
    # TODO: handle absolute paths?
    # TODO: add an option to disable file clobbering
    components = ("set", "animation", "frame", "sample")

    parser = argparse.ArgumentParser(add_help=False)
    parser.set_defaults(
        palette = os.path.join(os.path.dirname(sys.argv[0]), "Diamondus_2.pal"),
        log_level = logging.WARNING,
        exporter = json_extractor,
        settings = frozenset(("animation",))
    )

    class AdvOptAction(argparse._StoreAction):
        _used_opts = []

        def __call__(self, parser, namespace, values, option_string=None):
            type(self)._used_opts.append(option_string)
            super(AdvOptAction, self).__call__(parser, namespace, values, option_string)

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
                % (elt, globalDefaults["%s_defaults" % elt]["path_template"].replace("%", "%%"))
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

    args = parser.parse_args(argv)

    if args.exporter is legacy_extractor and AdvOptAction._used_opts:
        parser.error(
                "the following options are not available for the legacy extractor: %s"
                % ", ".join(frozenset(AdvOptAction._used_opts))
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
                path_formats[elt] = globalDefaults["%s_defaults" % elt]["path_template"]
            else:
                path_formats[elt] = fmt

        return args.exporter(animsfilename, args.output_folder, args.palette,
                path_formats=path_formats, settings_separate=args.settings)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
