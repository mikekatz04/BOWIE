import numpy as np
import multiprocessing as mp
import time


class ParallelContainer:
    """Run SNR calculation in parallel.

    Calculate in parallel using multiprocessing module.
    This can be easily adaptable to other parallel functions.

    Keyword Arguments:
        length (int): Number of binaries to process.
        num_processors (int or None, optional): If None, run on single processor.
            If -1, use ```multiprocessing.cpu_count()`` to determine cpus to use.
            Otherwise, this is the number of processors to use. Default is -1.
        num_splits (int, optional): Number of binaries to run for each process. Default is 1000.
        verbose (int, optional): Notify each time ``verbose`` processes finish.
            If -1, then no notification. Default is -1.
        timer (bool, optional): If True, time the parallel process. Default is False.

    Attributes:
        args (list of tuple): List of arguments to passes to the parallel function.
        Note: All kwargs above are stored as attributes.

    """

    def __init__(self, **kwargs):

        prop_defaults = {
            'num_processors': None,
            'num_splits': 1000,
            'verbose': -1,
            'timer': False,
        }

        required_kwarg_keys = ['length']

        for (prop, default) in prop_defaults.items():
                setattr(self, prop, kwargs.get(prop, default))

        for key in required_kwarg_keys:
            setattr(self, key, kwargs[key])

    def prep_parallel(self, binary_args, other_args):
        """Prepare the parallel calculations

        Prepares the arguments to be run in parallel.
        It will divide up arrays according to num_splits.

        Args:
            binary_args (list): List of binary arguments for input into the SNR function.
            other_args (tuple of obj): tuple of other args for input into parallel snr function.

        """
        if self.length < 100:
            raise Exception("Run this across 1 processor by setting num_processors kwarg to None.")
        if self.num_processors == -1:
            self.num_processors = mp.cpu_count()

        split_val = int(np.ceil(self.length/self.num_splits))
        split_inds = [self.num_splits*i for i in np.arange(1, split_val)]

        inds_split_all = np.split(np.arange(self.length), split_inds)

        self.args = []
        for i, ind_split in enumerate(inds_split_all):
            trans_args = []
            for arg in binary_args:
                try:
                    trans_args.append(arg[ind_split])
                except TypeError:
                    trans_args.append(arg)

            self.args.append((i, tuple(trans_args)) + other_args)
        return

    def run_parallel(self, para_func):
        """Run parallel calulation

        This will run the parallel calculation on self.num_processors.

        Args:
            para_func (obj): Function object to be used in parallel.

        Returns:
            (dict): Dictionary with parallel results.

        """
        if self.timer:
            start_timer = time.time()

        # for testing
        # check = parallel_snr_func(*self.args[10])
        # import pdb
        # pdb.set_trace()

        with mp.Pool(self.num_processors) as pool:
            print('start pool with {} processors: {} total processes.\n'.format(
                    self.num_processors, len(self.args)))

            results = [pool.apply_async(para_func, arg) for arg in self.args]
            out = [r.get() for r in results]
            out = {key: np.concatenate([out_i[key] for out_i in out]) for key in out[0].keys()}
        if self.timer:
            print("SNR calculation time:", time.time()-start_timer)
        return out
