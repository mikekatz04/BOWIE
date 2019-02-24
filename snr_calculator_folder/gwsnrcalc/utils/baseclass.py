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
        self.set_snr_prefactor()

        self.kwargs = kwargs

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
        keys = list(local_dict.keys())
        for key in keys:
            if isinstance(local_dict[key], np.ndarray):
                pass
            elif isinstance(local_dict[key], list):
                local_dict[key] = np.asarray(local_dict[key])

            # assume it is a float
            else:
                # turn it into length-1 array for mesh
                local_dict[key] = np.array([local_dict[key]])

        if 'output_keys' in self.__dict__:
            keys.insert(0, keys.pop(keys.index(self.output_keys[1])))
            keys.insert(0, keys.pop(keys.index(self.output_keys[0])))

        trans = np.meshgrid(*[local_dict[key] for key in keys])

        for key, arr in zip(keys, trans):
            setattr(self, key, arr.ravel())
        print('done')
        return

    def __run__(self, para_func):
        params = [getattr(self, key) for key in self.args_list]
        parallel_args = [getattr(self, key) for key in self.parallel_args
                         if key != 'num' and key != 'params']

        if len(params[0]) < 100 and self.num_processors != 1:
            Warning("Run this across 1 processor by setting num_processors kwarg to 0.")
            self.num_processors = 1

        if self.num_processors == 1:
            inputs = [0, params] + parallel_args
            out = para_func(*inputs)

        else:
            self.prep_parallel(params, parallel_args)
            out = self.run_parallel(para_func)

        if self.return_output == np.ndarray:
            for key, arr in out.items():
                setattr(self, key, arr)
            return

        elif self.return_output == dict:
            return out

        getattr(self, self.output_file_type + '_read_out')(out)

        return self.return_output

    def add_params(self, *args, **kwargs):
        if 'broadcast' in kwargs:
            self.set_broadcast(broadcast=kwargs['broadcast'])

        #for key, arg in zip(self.args_list, args):
        #    if isinstance(arg, float):


        if self.broadcast == 'mesh':
            self.meshgrid_and_set_attrs({key: value for key, value
                                         in zip(self.args_list, list(args))})

        else:
            self.broadcast_and_set_attrs({key: value for key, value
                                          in zip(self.args_list, list(args))})



        self.sources.not_broadcasted = False
        self.params_added = True
        return

    def set_signal_type(self, sig_type=['all']):
        """Set the signal type of interest.

        Sets the signal type for which the SNR is calculated.
        This means inspiral, merger, and/or ringdown.

        Args:
            sig_type (str or list of str): Signal type desired by user.
                Choices are `ins`, `mrg`, `rd`, `all` for circular waveforms created with PhenomD.
                If eccentric waveforms are used, must be `all`.

        """
        if isinstance(sig_type, str):
            sig_type = [sig_type]
        self.signal_type = sig_type
        return

    def set_snr_prefactor(self, factor=1.0):
        """Set the SNR multiplicative factor.

        This factor will be multpilied by the SNR, not SNR^2.
        This involves orientation, sky, and polarization averaging, as well
        as any factors for the configuration.

        For example, for LISA, this would be sqrt(2*16/5). The sqrt(2) is for
        a six-link configuration and the 16/5 represents the averaging factors.

        Args:
            factor (float): Factor to multiply SNR by for averaging.

        """
        self.prefactor = factor
        return

    def set_working_directory(self, wd='.'):
        """Set the WORKING_DIRECTORY variable.

        Sets the WORKING_DIRECTORY. The code will then use all paths as relative paths
        to the WORKING_DIRECTORY. In code default is current directory.

        Args:
            wd (str): Absolute or relative path to working directory.

        """
        self.WORKING_DIRECTORY = wd
        return

    def set_broadcast(self, broadcast='mesh'):
        """Set the broadcast variable.

        Sets the broadcast. This determines if you want to broadcast or
        meshgrid input arrays.

        Args:
            broadcast (str, optional): Broadcasting type. Options are `pure` or `mesh`.
                Default is `mesh`.

        """
        self.broadcast = broadcast
        return
