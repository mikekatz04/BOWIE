from .readout import FileReadOut
from .parallel import ParallelContainer


class BaseGenClass(FileReadOut, ParallelContainer):

    def __init__(self, return_input='data', print_input=False):

        # initialize defaults
        self.set_working_directory()
        self.set_broadcast()

        super(self, ParallelContainer).__init__()

        if return_input == 'file':
            super(self, FileReadOut).__init__()


    def _broadcast_and_set_attrs(self, local_dict):
        """Cast all inputs to correct dimensions.

        This method fixes inputs who have different lengths. Namely one input as
        an array and others that are scalara or of len-1.

        Raises:
            Value Error: Multiple length arrays of len>1

        """
        if 'self' in local_dict:
            del local_dict['self']

        self.remove_axis = False
        max_length = 0
        for key in local_dict:
            try:
                length = len(local_dict[key])
                if length > max_length:
                    max_length = length

            except TypeError:
                pass

        if max_length == 0:
            self.remove_axis = True
            for key in local_dict:
                setattr(self, key, np.array([local_dict[key]]))

        # check for bad length arrays
        else:
            for key in local_dict:
                try:
                    if len(local_dict[key]) < max_length and len(local_dict[key]) > 1:
                        raise ValueError("Casting parameters not correct."
                                         + " Need all at a maximum shape and the rest being"
                                         + "len-1 arrays or scalars")
                except TypeError:
                    pass

            # broadcast arrays
            for key in local_dict:
                try:
                    if len(local_dict[key]) == max_length:
                        setattr(self, key, local_dict[key])
                    elif len(local_dict[key]) == 1:
                        setattr(self, key, np.full((max_length,), local_dict[key][0]))
                except TypeError:
                    setattr(self, key, np.full((max_length,), local_dict[key]))
        return

    def _meshgrid_and_set_attrs(self, local_dict):
        """Mesh all inputs with meshgrid.

        This method combines all inputs in a grid.

        Raises:
            Value Error: Multiple length arrays of len>1

        """
        if 'self' in local_dict:
            del local_dict['self']

        self.remove_axis = False
        max_length = 0
        keys = local_dict.keys()
        for key in keys:
            if isinstance(local_dict[key], np.ndarray):
                pass
            elif isinstance(local_dict[key], list):
                local_dict[key] = np.asarray(local_dict[key])

            # assume it is a float
            else:
                # turn it into length-1 array for mesh
                local_dict[key] = np.array([local_dict[key]])

        trans = np.meshgrid([local_dict[key] for key in keys])
        for key, arr in zip(keys, trans):
            setattr(self, key, arr)

        return

    def run(self):
        self.prep_parallel()

    def set_working_directory(self, wd='.'):
        """Set the WORKING_DIRECTORY variable.

        Sets the WORKING_DIRECTORY. The code will then use all paths as relative paths
        to the WORKING_DIRECTORY. In code default is current directory.

        Args:
            wd (str): Absolute or relative path to working directory.

        """
        self.WORKING_DIRECTORY = wd
        return

    def set_broadcast(broadcast='pure'):
        """Set the broadcast_type variable.

        Sets the broadcast_type. This determines if you want to broadcast or
        meshgrid input arrays.

        Args:
            broadcast (str, optional): Broadcasting type. Options are `pure` or `mesh`.
                Default is `mesh`.

        """
        self.broadcast_type = broadcast
        return
