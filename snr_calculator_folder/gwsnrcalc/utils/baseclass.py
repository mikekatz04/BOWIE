from .readout import FileReadOut
from .parallel import ParallelContainer
from .waveforms import PhenomDWaveforms
import numpy as np


class BaseGenClass(FileReadOut, ParallelContainer):

    def __init__(self, **kwargs):

        prop_defaults = {
            'return_output': np.ndarray,
        }

        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        # initialize defaults
        self.set_working_directory()
        self.set_broadcast()

        ParallelContainer.__init__(self, **kwargs)

        if isinstance(self.return_output, str):
            print('Will return data in file {}'.format(self.return_output))
            FileReadOut.__init__(self, self.return_output, **kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def broadcast_and_set_attrs(self, local_dict):
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

    def meshgrid_and_set_attrs(self, local_dict):
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

        trans = np.meshgrid(*[local_dict[key] for key in keys])
        for key, arr in zip(keys, trans):
            setattr(self, key, arr.ravel())

        return

    def __run__(self, para_func):
        params = [getattr(self, key) for key in self.args_list]
        parallel_args = [getattr(self, key) for key in self.parallel_args
                         if key != 'num' and key != 'params']

        if len(params[0]) < 100 and self.num_processors != 0:
            Warning("Run this across 1 processor by setting num_processors kwarg to 0.")
            self.num_processors = 0

        if self.num_processors == 0:
            inputs = [0] + parallel_args + [0]
            out = para_func(*inputs)

        self.prep_parallel(params, parallel_args)

        out = self.run_parallel(para_func)

        if self.return_output == np.ndarray:
            for key, arr in out.items():
                setattr(self, key, arr)
            return

        elif self.return_output == dict:
            return out

        return self.return_output

    def set_working_directory(self, wd='.'):
        """Set the WORKING_DIRECTORY variable.

        Sets the WORKING_DIRECTORY. The code will then use all paths as relative paths
        to the WORKING_DIRECTORY. In code default is current directory.

        Args:
            wd (str): Absolute or relative path to working directory.

        """
        self.WORKING_DIRECTORY = wd
        return

    def set_broadcast(self, broadcast='pure'):
        """Set the broadcast variable.

        Sets the broadcast. This determines if you want to broadcast or
        meshgrid input arrays.

        Args:
            broadcast (str, optional): Broadcasting type. Options are `pure` or `mesh`.
                Default is `mesh`.

        """
        self.broadcast = broadcast
        return
