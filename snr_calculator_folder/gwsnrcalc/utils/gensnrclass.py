from .waveforms import PhenomDWaveforms, parallel_phenomd
from .gwwrappers import GWSNRWrapper
import inspect

def SNRGen(source_class=PhenomDWaveforms, snr_wrapper_class=GWSNRWrapper, **kwargs):
    class SNR(source_class, snr_wrapper_class):
        def __init__(self, **kwargs):
            self.instantiated = True
            self.params_added = False
            self.sources = source_class(**kwargs)
            self.args_list = inspect.getfullargspec(self.sources.__call__).args
            self.args_list.remove('self')
            snr_wrapper_class.__init__(self, **kwargs)

            for key in kwargs:
                setattr(self, key,  kwargs[key])
            self.parallel_func_name = self.sources.instantiate_parallel_func()
            self.parallel_args = inspect.getfullargspec(globals()[self.parallel_func_name]).args
    return SNR(**kwargs)
