# coding=utf-8

import copy
import json
import os
import re
from contextlib import contextmanager
from loguru import logger
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

from .utils import (
    CONFIG_NAME, 
    WEIGHTS_NAME, 
    cached_path, hf_bucket_url
)

try:
    from torch.hub import _get_torch_home
    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(os.getenv('TORCH_HOME', os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))


_init_weights = True

@contextmanager
def no_init_weights(_enable=True):
    """
    Context manager to globally disable weight initialization to speed up loading large models.
    """
    global _init_weights
    if _enable:
        _init_weights = False
    try:
        yield
    finally:
        _init_weights = True


class BaseConfig:
    """
    Base configuration class. Handles methods for loading/saving configurations.
    """
    model_type: str = ""
    
    def __init__(self, **kwargs):
        # Attributes with defaults
        self.output_hidden_states = kwargs.pop("output_hidden_states", False)
        
        # Fine-tuning task arguments
        self.architectures = kwargs.pop("architectures", None)
        self.id2label = kwargs.pop("id2label", None)
        self.label2id = kwargs.pop("label2id", None)
        if self.id2label is not None:
            # Keys are always strings in JSON so convert ids to int here.
            self.id2label = dict((int(key), value) for key, value in self.id2label.items())
        
        # regression / multi-label classification
        self.problem_type = kwargs.pop("problem_type", None)
        allowed_problem_types = ("regression", "single_label_classification", "multi_label_classification")
        if self.problem_type is not None and self.problem_type not in allowed_problem_types:
            raise ValueError("The config parameter `problem_type` was not understood: received {} but only "
                             "'regression', 'single_label_classification' and 'multi_label_classification' are valid.".format(self.problem_type))

    @property
    def num_labels(self) -> int:
        """
        The number of labels for classification models.
        """
        if self.id2label is not None:
            return len(self.id2label)
        return 2   
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs):
        """
        Constructs a `BaseConfig` from a Python dictionary of parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        config = cls(**config_dict)
        
        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike], **kwargs):
        """
        Constructs a `BaseConfig` from the path to a JSON file of parameters.
        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)
        return cls.from_dict(config_dict, **kwargs)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        """
        Constructs a `BaseConfig` from a pretrained model configuration.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        if cache_dir is None:
            cache_dir = os.path.join(torch_cache_home, "bertorch")
            cache_dir = os.path.join(cache_dir, pretrained_model_name_or_path.replace("/", "_"))
        os.makedirs(cache_dir, exist_ok=True)
        
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            config_json_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        elif os.path.isfile(pretrained_model_name_or_path):
            config_json_file = pretrained_model_name_or_path
        else:
            config_url = hf_bucket_url(pretrained_model_name_or_path, CONFIG_NAME)

            # Load `config.json` from URL
            try:
                config_json_file = cached_path(config_url, filename=CONFIG_NAME, cache_dir=cache_dir)
            except EnvironmentError as err:
                logger.error(err)
                msg = (
                    "Can't load config.json for '{}'. Make sure that: '{}' is a correct model identifier "
                    "listed on https://huggingface.co/models".format(pretrained_model_name_or_path, pretrained_model_name_or_path)
                )
                raise EnvironmentError(msg)

        return cls.from_json_file(config_json_file, **kwargs)
    
    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded 
        using the BaseConfig.from_pretrained class method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError("Provided path ({}) should be a directory, not a file".format(save_directory))
        
        os.makedirs(save_directory, exist_ok=True)
        output_config_file = os.path.join(save_directory, CONFIG_NAME)
        
        self.to_json_file(output_config_file)
        #logger.info("Configuration saved in {}".format(output_config_file))
               
    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. 
        """
        output = copy.deepcopy(self.__dict__)
        keys = list(output.keys())
        for key in keys:
            value = output[key]
            if (value is None 
                or key in ("architectures", "output_hidden_states", "problem_type")):
                del output[key]
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())
    

class PreTrainedModel(nn.Module):
    """
    Base class for all models. `PreTrainedModel` takes care of storing the configuration of the models 
    and handles methods for loading and downloading models.
    """
    config_class = None
    base_model_prefix = ""
    # a list of re pattern of tensor names to ignore from the model when loading the model weights
    # (and avoid unnecessary warnings).
    _keys_to_ignore_on_load_missing = None
    # a list of re pattern of tensor names to ignore from the weights when loading the model weights
    # (and avoid unnecessary warnings).
    _keys_to_ignore_on_load_unexpected = None

    def __init__(self, config: BaseConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, BaseConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`BaseConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config
    
    def init_weights(self):
        if _init_weights:
            # Initialize weights
            self.apply(self._init_weights)
      
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        """
        Instantiate a pretrained pytorch model from a pre-trained model name or path. 
        """
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        cache_dir = kwargs.pop("cache_dir", None)
        _fast_init = kwargs.pop("_fast_init", True)
        
        # cache folder when downloading a pretrained pytorch model from https://huggingface.co/
        if cache_dir is None:
            cache_dir = os.path.join(torch_cache_home, "bertorch")
            cache_dir = os.path.join(cache_dir, pretrained_model_name_or_path.replace("/", "_"))
        os.makedirs(cache_dir, exist_ok=True)
        
        # pytorch model path
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            model_bin_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
            config_json_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        elif os.path.isfile(pretrained_model_name_or_path):
            model_bin_file = pretrained_model_name_or_path
            config_json_file = None
        else:
            model_url = hf_bucket_url(pretrained_model_name_or_path, WEIGHTS_NAME)
            config_url = hf_bucket_url(pretrained_model_name_or_path, CONFIG_NAME)
            
            try:
                # Load `pytorch_model.bin` and `config.json` from URL
                model_bin_file = cached_path(model_url, filename=WEIGHTS_NAME, cache_dir=cache_dir)
                config_json_file = cached_path(config_url, filename=CONFIG_NAME, cache_dir=cache_dir)
            except EnvironmentError as err:
                logger.error(err)
                msg = (
                    "Can't load weights for '{}'. Make sure that: '{}' is a correct model identifier "
                    "listed on https://huggingface.co/models".format(pretrained_model_name_or_path, pretrained_model_name_or_path)
                )
                raise EnvironmentError(msg)
        
        if not os.path.isfile(model_bin_file):
            raise ValueError("pytorch model file does not exist.")

        # Load config if we don't provide a configuration
        if not isinstance(config, BaseConfig):
            if config_json_file is None:
                raise ValueError("missing config parameter.")
            if not os.path.isfile(config_json_file):
                raise ValueError("config json file does not exist.")
            config, model_kwargs = cls.config_class.from_json_file(config_json_file, return_unused_kwargs=True, **kwargs)
        else:
            model_kwargs = kwargs
        
        logger.info("loading weights from {}".format(model_bin_file))
        
        with no_init_weights(_enable=_fast_init):
            model = cls(config, *model_args, **model_kwargs)
        
        if state_dict is None:
            try:
                state_dict = torch.load(model_bin_file, map_location="cpu")
            except Exception:
                raise OSError(
                    "Unable to load weights from pytorch checkpoint file for {} at {}".format(
                    pretrained_model_name_or_path, model_bin_file)
                )

        model = cls._load_state_dict_into_model(model, state_dict, _fast_init=_fast_init)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        return model

    @classmethod
    def _load_state_dict_into_model(cls, model, state_dict, _fast_init=True):
        
        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)
        
        # Retrieve missing & unexpected_keys
        expected_keys = list(model.state_dict().keys())
        loaded_keys = list(state_dict.keys())
        prefix = model.base_model_prefix
        
        has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
        expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)

        # key re-naming operations are never done on the keys
        # that are loaded, but always on the keys of the newly initialized model
        remove_prefix = not has_prefix_module and expects_prefix_module
        add_prefix = has_prefix_module and not expects_prefix_module

        if remove_prefix:
            expected_keys = [".".join(s.split(".")[1:]) if s.startswith(prefix) else s for s in expected_keys]
        elif add_prefix:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        # Some models may have keys that are not in the state by design, removing them before needlessly warning
        # the user.
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]
        
        if _fast_init:
            # retrieve unintialized modules and initialize
            unintialized_modules = model.retrieve_modules_from_names(
                missing_keys, add_prefix=add_prefix, remove_prefix=remove_prefix
            )
            for module in unintialized_modules:
                model._init_weights(module)
        
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        error_msgs = []
        
        def load(module: nn.Module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
            module._load_from_state_dict(*args)
            
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")
        
        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if not hasattr(model, cls.base_model_prefix) and has_prefix_module:
            start_prefix = cls.base_model_prefix + "."
        if hasattr(model, cls.base_model_prefix) and not has_prefix_module:
            model_to_load = getattr(model, cls.base_model_prefix)

        load(model_to_load, prefix=start_prefix)
        
        if len(missing_keys) > 0:
            logger.warning("Some weights of {} were not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.warning("Some weights from pretrained model were not used when initializing {}: {}".format(
                model.__class__.__name__, unexpected_keys))    
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)))
        
        return model
    
    def retrieve_modules_from_names(self, names, add_prefix=False, remove_prefix=False):
        module_keys = set([".".join(key.split(".")[:-1]) for key in names])
        module_keys = module_keys.union(set([".".join(key.split(".")[:-2]) for key in names if key[-1].isdigit()]))
        
        retrieved_modules = []
        # retrieve all modules that has at least one missing weight name
        for name, module in self.named_modules():
            if remove_prefix:
                name = ".".join(name.split(".")[1:]) if name.startswith(self.base_model_prefix) else name
            elif add_prefix:
                name = ".".join([self.base_model_prefix, name]) if len(name) > 0 else self.base_model_prefix

            if name in module_keys:
                retrieved_modules.append(module)
        
        return retrieved_modules
