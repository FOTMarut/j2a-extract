import itertools

class PropertiesCounter(object):
    __slots__ = ["_properties"]

    def __init__(self):
        self._properties = {}

    def update(self, d):
        '''
        Add to the record all key-value pairs from a mapping type

        Parameters
        ----------
        d : mapping type with an items() method
            Input mapping

        '''
        for k,v in d.items():
            prop = self._properties.setdefault(k, {})
            prop[v] = prop.get(v, 0) + 1

    def update_single(self, key, value):
        '''
        Add a single key-value pair to the record

        Parameters
        ----------
        key : object
            The key
        value : object
            The value
        '''
        prop = self._properties.setdefault(key, {})
        prop[value] = prop.get(value, 0) + 1

    def _get_modes_iter(self, threshold):
        for prop, values in self._properties.items():
            if values:
                total = sum(values.values())
                best_val, num_occurrences = max(values.items(), key = lambda t : t[1])
                if num_occurrences > threshold:
                    yield (prop, best_val)

    def get_modes(self, threshold = -1):
        '''
        Get a dict mapping each key to its most common value

        Prameters
        ---------
        threshold : int
            When querying modes, only values occurring more than this number of
            times are returned; default: -1 (return all modes)

        Returns
        -------
        m : dict
            New dict mapping a key k to a value v if and only if, among all
            invocations of `update` and `update_single`, v was the value most
            commonly associated with k and the `(k, v)` pair was found strictly
            more than threshold` number of times.
        '''
        return dict(self._get_modes_iter(threshold))

class MetadataInterface(object):
    '''
    MetadataInterface(name, md_generator = lambda obj : {},
                      base_idx_function = lambda obj : None, children = ())

    Facility to manipulate J2A metadata dictionaries
    Every instance of a MetadataInterface represents a type in the metadata
    tree.

    Should be constructed with its children already constructed, as opposed
    to setting the children field afterwards.

    Parameters
    ----------
    name : str
        Name of the represented type
    md_generator : callable returning a dict, optional
        A callable that takes a single argument, the object to serialize,
        and returns a dict the object's global properties
    base_idx_function : callable returning an int, optional
        A callable that takes a single argument, the parent object, and
        returns the base index pertaining to the instance's type under the
        parent object
    children : tuple of MetadataInterfaces
        The subtypes

    Attributes
    ----------
    name : str
        Name of the represented type
    sub_name : str
        Key used in the metadata tree to identify a collection of such type
        instances
    defs_name : str
        Key used in the metadata tree to identify a dict of defaults for
        this type
    md_generator : callable returning a dict
        The omonymous argument given at object construction
    base_idx_function : callable returning an int
        The omonymous argument given at object construction
    children : tuple of MetadataInterfaces
        The omonymous argument given at object construction
    descendant_defs_names : tuple of strings
        A tuple with the names of all defaults for (indirect) subtypes
    '''
    __slots__ = ["name", "sub_name", "defs_name", "base_idx_function", "md_generator", "children", "descendant_defs_names"]

    def __init__(self, name,
                 md_generator = lambda obj : {},
                 base_idx_function = lambda obj : None,
                 children = ()
                 ):
        # "child" refers to a child type, not a single element
        # (e.g. "set"'s children are "animation" and "sample")
        self.name              = name
        self.sub_name          = name + "s"
        self.defs_name         = name + "_defaults"
        self.md_generator      = md_generator
        self.base_idx_function = base_idx_function
        self.children          = tuple(children)
        self.descendant_defs_names = tuple( itertools.chain.from_iterable(
            (child_md_iface.defs_name,) + child_md_iface.descendant_defs_names
            for child_md_iface in self.children
            ) )

    @staticmethod
    def _generate_defaults(elements_md, main_kind, sub_kinds = (), threshold_ratio = 0.5):
        main_defaults_counter = PropertiesCounter()
        sub_defaults_counters = dict( (kind, PropertiesCounter()) for kind in sub_kinds)

        # Count the number of occurrences for each property
        for elt_md in elements_md:
            for k,v in elt_md.items():
                if k in sub_kinds:
                    sub_defaults_counters[k].update(v)
                elif not isinstance(v, dict):
                    main_defaults_counter.update_single(k, v)

        # Collect for each property the most common cases
        threshold = threshold_ratio * len(elements_md)
        sub_metadata = dict(
                (kind, counter.get_modes(threshold))
                for kind,counter in sub_defaults_counters.items()
                )
        main_metadata = main_defaults_counter.get_modes(threshold)

        sub_metadata[main_kind] = main_metadata
        return sub_metadata

    def _recursive_apply_impl(self, md, func_before, func_after, child_accessor, *pargs, **kwargs):
        pargs = func_before(self, md, *pargs, **kwargs)
        if pargs is None:
            pargs = ()
        for child_md_iface in self.children:
            sub = md[child_md_iface.sub_name]
            for i in range(sub['count']):
                child_md_iface._recursive_apply_impl(
                        child_accessor(sub, i, {}), func_before, func_after, child_accessor, *pargs, **kwargs
                        )
        return func_after(self, md, *pargs, **kwargs)

    def recursive_apply(self, metadata, *pargs, **kwargs):
        '''
        Recursively apply the given functions to `metadata` and all its
        subobjects.

        Parameters
        ----------
        func_before : callable, optional
            (metadata_interface, metadata, *pargs, **kwargs) -> None or tuple
            Invoked on each object before recurring into its children; the
            return value is taken as a tuple of extra positional arguments for
            invocations of `func_before` and `func_after` on the children.
            The given metadata interface is always coherent with the given
            element.
            A return value of `None` is interpreted as `()`.
            The top-level invocation's extra positional arguments are set to
            this function's `pargs`.
            The keyword arguments are always set to this function's `kwargs`.
        func_after : callable, optional
            (metadata_interface, metadata, *pargs, **kwargs) -> object
            Like `func_before`, but it's invoked after recurring into an
            object's children and has no restriction or special meaning to the
            return value
            The top-level invocation's return value is returned by this
            function.
        child_accessor : callable, optional
            (d: dict, i: int, def: dict) -> dict
            Used to access a child from its containing dict, should implement:
                `d[i] if i in d else def`
            Default is dict.get
            It may be useful to override in order to insert any missing
            elements, e.g. with dict.setdefault

        Returns
        -------
        x : object
            The result of the top-level invocation of `func_after`, or `None`
        '''
        func_before    = kwargs.pop("func_before",    lambda self, metadata, *pargs, **kwargs : pargs)
        func_after     = kwargs.pop("func_after",     lambda self, metadata, *pargs, **kwargs : None)
        child_accessor = kwargs.pop("child_accessor", dict.get)
        return self._recursive_apply_impl(
                metadata, func_before, func_after, child_accessor, *pargs, **kwargs
                )

    @staticmethod
    def _iterable_to_dict_items(seq, base_idx = None):
        if base_idx is not None:
            yield ("base_idx", base_idx)
        yield ("count", len(seq))
        for i,v in enumerate(seq):
            if v:
                yield (i, v)

    def _generate_metadata(self, obj, generate_defaults = True):
        metadata = self.md_generator(obj)
        for child_md_iface in self.children:
            child_md_list = [
                    child_md_iface._generate_metadata(child, generate_defaults)
                    for child in getattr(obj, child_md_iface.sub_name)
                    ]
            if generate_defaults:
                metadata.update( self._generate_defaults(
                    child_md_list, child_md_iface.defs_name, child_md_iface.descendant_defs_names
                    ) )
            base_idx = child_md_iface.base_idx_function(obj)
            metadata[child_md_iface.sub_name] = \
                    dict(self._iterable_to_dict_items(child_md_list, base_idx = base_idx))
        return metadata

    @staticmethod
    def _strip_matching_defaults(d, defaults):
        for k,v in defaults.items():
            if k in d and d[k] == v:
                del d[k]
        return d

    def _suppress_redundant_defaults(self, md, defaults):
        '''
        PRE:
        At this point the metadata structure should be complete, i.e. at all levels:
            all(defs_name in md for defs_name in self.descendant_defs_names)
            all(
                i in md[child_md_iface.sub_name]
                for child_md_iface in self.children
                for i in range(md[child_md_iface.sub_name]["count"])
            )
        and also all the properties of the object should be set.
        However, the sub defaults must NOT necessarily be complete (e.g. insufficient value dominance).

        It is required that for each lone child there is a corresponding default in the parent
        with the exact same settings, which is the behavior of _generate_defaults

        defaults must be exhaustive

        DURING:
        ONLY change md and its descendants.
        '''
        self._strip_matching_defaults(md, defaults[self.defs_name])
        for defs_name in self.descendant_defs_names:
            if not self._strip_matching_defaults(md[defs_name], defaults[defs_name]):
                del md[defs_name]

        # Remove defaults applying to only children
        for child_md_iface in self.children:
            sub = md[child_md_iface.sub_name]
            sub_count = sub["count"]
            if sub_count <= 1:
                for defs_name in (child_md_iface.defs_name,) + child_md_iface.descendant_defs_names:
                    md.pop(defs_name, None)

        # Generate new defaults
        new_defaults = dict(
                (defs_name, dict(defaults[defs_name], **md.get(defs_name, {})))
                for defs_name in self.descendant_defs_names
                )

        # Recursion into each child
        for child_md_iface in self.children:
            sub = md[child_md_iface.sub_name]
            sub_count = sub["count"]
            for i in range(sub_count):
                if not child_md_iface._suppress_redundant_defaults(sub[i], new_defaults):
                    del sub[i]

        return md

    def generate_metadata(self, obj, defaults = None):
        '''
        Produces and returns metadata associated with the given object
        (including any sub-objects).

        Parameters
        ----------
        obj : object
            An object of type appropriate to the MetadataInterface instance
        defaults : dict, optional
            If not given or `None`, the bare metadata structure will be
            returned.
            If given, it must be a dict of defaults for all metadata types in
            the tree; appropriate defaults will be produced at each intermediate
            level and redundancy will be removed using this argument as global
            defaults.

        Returns
        -------
        md : dict
            The generated metadata
        '''
        md = self._generate_metadata(obj, defaults is not None)
        if defaults is not None:
            self._suppress_redundant_defaults(md, defaults)
        return md

    def _validate_metadata_impl(self, metadata, keys_mapping):
        dict_keys, prop_keys = keys_mapping[self]
        for k,v in metadata.items():
            if isinstance(v, dict):
                if k not in dict_keys:
                    raise ValueError(
                            "'%s' object has extraneous submapping '%s'; expected one of: %s" % (
                                self.name, k, ", ".join(map(repr, dict_keys))
                                ) )

            elif not k in prop_keys:
                raise ValueError(
                        "'%s' object has extraneous property '%s'; expected one of: %s" % (
                            self.name, k, ", ".join(map(repr, prop_keys))
                            ) )

        for child_md_iface in self.children:
            try:
                sub = metadata[child_md_iface.sub_name]
            except KeyError:
                raise ValueError(
                        "Missing %r collection in %r object" % (
                            child_md_iface.sub_name, self.name
                            ) )

            try:
                sub_count = sub["count"]
            except KeyError:
                raise ValueError(
                        "%r collection in %r object missing required field 'count'" % (
                            child_md_iface.sub_name, self.name
                            ) )

            extra_keys = set(sub) - set(itertools.chain(("base_idx", "count"), range(sub_count)))
            if extra_keys:
                raise ValueError(
                        "%r collection in %r object has extraneous element(s) %s" % (
                            child_md_iface.sub_name, self.name, ", ".join(map(repr, extra_keys))
                            ) )


    def validate_metadata(self, metadata, defaults):
        '''
        Performs an all-round validation of metadata

        Checks that there are no extraneous keys in metadata, distinguishing
        between those mapped to dicts from the others.
        Raises ValueError if it finds an error.

        Parameters
        ----------
        metadata : dict
            The metadata to validate
        defaults : dict
            Used as a template to get allowed keys for each metadata type
        '''
        def get_defaults_key_mapping(md_iface):
            yield (md_iface, (
                set( itertools.chain(
                    md_iface.descendant_defs_names,
                    (child_md_iface.sub_name for child_md_iface in md_iface.children)
                    ) ),
                set(defaults.get(md_iface.defs_name, {}))
                ) )
            for child_md_iface in md_iface.children:
                for t in get_defaults_key_mapping(child_md_iface):
                    yield t

        # Maps each metadata interfaces to a tuple (set of keys mapped to dicts, set of property keys)
        keys_mapping = dict(get_defaults_key_mapping(self))
        self.recursive_apply(
                metadata,
                func_before = type(self)._validate_metadata_impl,
                keys_mapping = keys_mapping
                )

    def _propagate_defaults_impl(self, metadata, defaults):
        '''
        PRE:
        `defaults` must contain appropriate defaults for all types included in metadata
        POST:
        No defaults whatsoever in `metadata` or any of the descendants
        All properties of each element are explicitly set
        All children are explicitly present
        '''
        # Apply defaults
        for k,v in defaults[self.defs_name].items():
            metadata.setdefault(k, v)

        if self.children:
            # Generate new defaults on top of the parent's ones
            new_defaults = dict((k, defaults[k].copy()) for k in self.descendant_defs_names)
            for defs_name in self.descendant_defs_names:
                new_defaults[defs_name].update(metadata.pop(defs_name, {}))

            return (new_defaults,)

    def propagate_defaults(self, metadata, defaults):
        '''
        Remove all defaults from `metadata` and apply them to the appropriate
        objects.

        Parameters
        ----------
        metadata : dict
            Metadata associated to an object whose type is coherent with that of
            the MetadataInterface instance
        defaults : dict
            Dictionary of global defaults for all metadata types in the metadata
            tree

        Returns
        -------
        metadata : dict
            The input metadata, after in-place update
        '''
        return self.recursive_apply(
                metadata,
                defaults,
                func_before = type(self)._propagate_defaults_impl,
                child_accessor = dict.setdefault
                )


def generate_frame_metadata(frame):
    return {
        "hotspot" : frame.origin,
        "coldspot": frame.coldspot,
        "gunspot" : frame.gunspot,
        "tagged"  : frame.tagged,
        "mask"    : frame.mask is not None and any(frame.mask)  # FIXME: export mask if it's non-trivial
    }

def generate_sample_metadata(sample):
    loop_start, loop_end, loop_bidi = sample.loop or (0, None, False)
    return {
        "volume"    : sample.volume,
        "loop_start": loop_start,
        "loop_end"  : loop_end,
        "loop_bidi" : loop_bidi
    }

generate_animation_metadata = lambda animation : {"fps": animation.fps}

j2a_metadata_interface = MetadataInterface( "J2A", children = (
    MetadataInterface( "set", children = (
        MetadataInterface( "animation", md_generator = generate_animation_metadata,
                                        children = (
            MetadataInterface( "frame", md_generator = generate_frame_metadata),
        ) ),
        MetadataInterface( "sample", md_generator = generate_sample_metadata,
                                     base_idx_function = lambda set_ : set_.samplesbaseindex )
    ) ),
) )


globalDefaults = dict(
    J2A_defaults = {
        "separate_settings": ("animations",)
    },
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
