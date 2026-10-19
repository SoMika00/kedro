"""This module provides ``kedro.config`` with the functionality to load one
or more configuration files of yaml or json type from specified paths through OmegaConf.
"""

from __future__ import annotations

import io
import logging
import mimetypes
import re
import typing
from collections.abc import Callable, Iterable, KeysView
from enum import Enum, auto
from pathlib import Path
from typing import Any

import fsspec
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError, UnsupportedInterpolationType
from omegaconf.resolvers import oc
from yaml.parser import ParserError
from yaml.scanner import ScannerError

from kedro.config.abstract_config import AbstractConfigLoader, MissingConfigException
from kedro.utils import CLOUD_PROTOCOLS, HTTP_PROTOCOLS, _parse_filepath

_config_logger = logging.getLogger(__name__)

_NO_VALUE = object()

# Marks a dataset's `type` field as resolved through the `runtime_params:`
# resolver, e.g. `type: "${runtime_params:dataset.type}"`.
_RUNTIME_PARAMS_MARKER = "runtime_params:"
# Extracts the variable path passed to the `runtime_params:` resolver, e.g.
# "dataset.type" from `${runtime_params:dataset.type}` or
# `${runtime_params:dataset.type, 'MemoryDataset'}`.
_RUNTIME_PARAMS_VARIABLE_RE = re.compile(r"\$\{runtime_params:\s*([^,}]+)")


class MergeStrategies(Enum):
    SOFT = auto()
    DESTRUCTIVE = auto()


MERGING_IMPLEMENTATIONS = {
    MergeStrategies.SOFT: "_soft_merge",
    MergeStrategies.DESTRUCTIVE: "_destructive_merge",
}


class OmegaConfigLoader(AbstractConfigLoader):
    """Recursively scan directories (config paths) contained in ``conf_source`` for
    configuration files with a ``yaml``, ``yml`` or ``json`` extension, load and merge
    them through ``OmegaConf`` (https://omegaconf.readthedocs.io/)
    and return them in the form of a config dictionary.

    The first processed config path is the ``base`` directory inside
    ``conf_source``. The optional ``env`` argument can be used to specify a
    subdirectory of ``conf_source`` to process as a config path after ``base``.

    When the same top-level key appears in any two config files located in
    the same (sub)directory, a ``ValueError`` is raised.

    When the same key appears in any two config files located in different
    (sub)directories, the last processed config path takes precedence
    and overrides this key and any sub-keys.

    You can access the different configurations as follows:
    ``` python
    import logging.config
    from kedro.config import OmegaConfigLoader
    from kedro.framework.project import settings

    conf_path = str(project_path / settings.CONF_SOURCE)
    conf_loader = OmegaConfigLoader(conf_source=conf_path, env="local")

    conf_catalog = conf_loader["catalog"]
    conf_params = conf_loader["parameters"]
    ```

    ``OmegaConf`` supports variable interpolation in configuration
    https://omegaconf.readthedocs.io/en/2.2_branch/usage.html#merging-configurations. It is
    recommended to use this instead of yaml anchors with the ``OmegaConfigLoader``.

    This version of the ``OmegaConfigLoader`` does not support any of the built-in ``OmegaConf``
    resolvers. Support for resolvers might be added in future versions.

    To use this class, change the setting for the `CONFIG_LOADER_CLASS` constant
    in `settings.py`.

    Example:
    ``` python
    # in settings.py
    from kedro.config import OmegaConfigLoader

    CONFIG_LOADER_CLASS = OmegaConfigLoader
    ```
    """

    def __init__(  # noqa: PLR0913
        self,
        conf_source: str | Path,
        env: str | None = None,
        runtime_params: dict[str, Any] | None = None,
        *,
        config_patterns: dict[str, list[str]] | None = None,
        base_env: str | None = None,
        default_run_env: str | None = None,
        custom_resolvers: dict[str, Callable | tuple[Callable, dict[str, Any]]]
        | None = None,
        merge_strategy: dict[str, str] | None = None,
        ignore_hidden: bool = True,
        restrict_runtime_params_type_selection: bool = False,
    ):
        if isinstance(conf_source, Path):
            conf_source = str(conf_source)
        """Instantiates a ``OmegaConfigLoader``.

        Args:
            conf_source: Path to use as root directory for loading configuration.
                This can be a local filesystem path or a remote URL with protocol
                (e.g., s3://, gs://, etc.)
            env: Environment that will take precedence over base.
            runtime_params: Extra parameters passed to a Kedro run.
            config_patterns: Regex patterns that specify the naming convention for configuration
                files so they can be loaded. Can be customised by supplying config_patterns as
                in `CONFIG_LOADER_ARGS` in `settings.py`.
            base_env: Name of the base environment. When the ``OmegaConfigLoader`` is used directly
                this defaults to `None`. Otherwise, the value will come from the `CONFIG_LOADER_ARGS` in the project
                settings, where base_env defaults to `"base"`.
                This is used in the `conf_paths` property method to construct
                the configuration paths.
            default_run_env: Name of the default run environment. When the ``OmegaConfigLoader`` is used directly
                this defaults to `None`. Otherwise, the value will come from the `CONFIG_LOADER_ARGS` in the project
                settings, where default_run_env defaults to `"local"`.
                Can be overridden by supplying the `env` argument.
            custom_resolvers: A dictionary of custom resolvers to be registered. For more information,
                see here: https://omegaconf.readthedocs.io/en/2.3_branch/custom_resolvers.html#custom-resolvers
                A resolver can be given as a `(callable, kwargs)` tuple to pass keyword arguments such as
                `use_cache` or `replace` on to `OmegaConf.register_new_resolver()`.
            merge_strategy: A dictionary that specifies the merging strategy for each configuration type.
                The accepted merging strategies are `soft` and `destructive`. Defaults to `destructive`.
            ignore_hidden: A boolean flag that determines whether hidden files and directories should be
                ignored when loading configuration files. If True, ignore hidden files and files in hidden directories.
            restrict_runtime_params_type_selection: When True, a catalog dataset's `type` is not
                allowed to resolve through the `runtime_params:` resolver (e.g. `type:
                "${runtime_params:dataset.type}"`) at all -- `runtime_params` must not be able to
                choose which class gets dynamically imported and instantiated. Defaults to False,
                since `runtime_params` is normally supplied by a trusted caller (e.g. `kedro run
                --params`). Callers that construct this loader with `runtime_params` sourced from
                an untrusted caller (e.g. an HTTP request body) must pass True.
        """
        self.ignore_hidden = ignore_hidden
        self.restrict_runtime_params_type_selection = (
            restrict_runtime_params_type_selection
        )
        self.base_env = base_env or ""
        self.default_run_env = default_run_env or ""
        self.merge_strategy = merge_strategy or {}
        self._globals_oc: DictConfig | None = None
        self._runtime_params_oc: DictConfig | None = None
        # Paths where runtime_params actually resolved, not a resolver default.
        self._runtime_params_hits: set[str] = set()
        self.config_patterns = {
            "catalog": ["catalog*", "catalog*/**", "**/catalog*"],
            "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
            "credentials": ["credentials*", "credentials*/**", "**/credentials*"],
            "globals": ["globals.yml"],
        }
        self.config_patterns.update(config_patterns or {})

        # Deactivate oc.env built-in resolver for OmegaConf
        OmegaConf.clear_resolver("oc.env")
        # Register user provided custom resolvers
        self._custom_resolvers = custom_resolvers
        if custom_resolvers:
            self._register_new_resolvers(custom_resolvers)
        # Register globals resolver
        self._register_globals_resolver()

        # Setup file system and protocol
        self._fs, self._protocol = self._initialise_filesystem_and_protocol(conf_source)

        # Store remote root path if using cloud protocol
        if self._protocol in CLOUD_PROTOCOLS or self._protocol in HTTP_PROTOCOLS:
            options = _parse_filepath(conf_source)
            self._remote_root_path = options["path"].rstrip("/")

        super().__init__(
            conf_source=conf_source,
            env=env,
            runtime_params=runtime_params,
        )
        try:
            self._globals = self["globals"]
        except MissingConfigException:
            self._globals = {}

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "globals":
            # Update the cached value at self._globals since it is used by the globals resolver
            self._globals = value
        super().__setitem__(key, value)

    def __getitem__(self, key: str) -> dict[str, Any]:
        """Get configuration files by key, load and merge them, and
        return them in the form of a config dictionary.

        Args:
            key: Key of the configuration type to fetch.

        Raises:
            KeyError: If key provided isn't present in the config_patterns of this
                ``OmegaConfigLoader`` instance.
            MissingConfigException: If no configuration files exist matching the patterns
                mapped to the provided key.

        Returns:
            Dict[str, Any]:  A Python dictionary with the combined
                configuration from all configuration files.
        """
        # Allow bypassing of loading config from patterns if a key and value have been set
        # explicitly on the ``OmegaConfigLoader`` instance.

        # Re-register runtime params resolver in case it was previously deactivated
        self._register_runtime_params_resolver()

        if key in self:
            return super().__getitem__(key)  # type: ignore[no-any-return]

        if key not in self.config_patterns:
            raise KeyError(
                f"No config patterns were found for '{key}' in your config loader"
            )

        base_configs, env_configs, base_path, env_path, processed_files = (
            self._read_raw_config(key)
        )
        return self._resolve_from_raw_config(
            key, base_configs, env_configs, base_path, env_path, processed_files
        )

    def _read_raw_config(
        self, key: str
    ) -> tuple[dict[Path, DictConfig], dict[Path, DictConfig], str, str, set[Path]]:
        """Read raw per-file configs from base and env dirs for ``key``.

        Split out so callers can cache the parse and only re-pay the
        resolve cost per request.
        """
        patterns = [*self.config_patterns[key]]
        read_environment_variables = key == "credentials"

        processed_files: set[Path] = set()
        base_path = self._get_conf_env_path(self.base_env)
        base_configs = self._read_dir_configs(
            base_path, patterns, key, processed_files, read_environment_variables
        )

        run_env = self.env or self.default_run_env
        env_path = self._get_conf_env_path(run_env)
        if run_env == self.base_env:
            return base_configs, {}, base_path, env_path, processed_files

        env_configs = self._read_dir_configs(
            env_path, patterns, key, processed_files, read_environment_variables
        )
        return base_configs, env_configs, base_path, env_path, processed_files

    def _get_conf_env_path(self, env: str) -> str:
        """Get the path to the configuration environment directory."""
        if not env:
            return str(self.conf_source)
        return str(Path(self.conf_source) / env)

    def _read_dir_configs(
        self,
        conf_path: str,
        patterns: list[str],
        key: str,
        processed_files: set[Path],
        read_environment_variables: bool,
    ) -> dict[Path, DictConfig]:
        """Read and parse all config files in ``conf_path`` matching ``patterns``."""
        if not self._fs.exists(conf_path):
            if key == "globals":
                return {}
            raise MissingConfigException(
                f"Given configuration path either does not exist "
                f"or is not a valid directory: {conf_path}"
            )

        configs: dict[Path, DictConfig] = {}
        for filepath in self._get_config_files(conf_path, patterns):
            if filepath in processed_files:
                continue
            processed_files.add(filepath)
            configs[filepath] = self._load_config_file(
                filepath, read_environment_variables
            )
        return configs

    def _get_config_files(self, conf_path: str, patterns: list[str]) -> list[Path]:
        """Return the list of config files under ``conf_path`` matching ``patterns``."""
        if self._protocol in CLOUD_PROTOCOLS or self._protocol in HTTP_PROTOCOLS:
            return self._get_remote_config_files(conf_path, patterns)

        conf_path_obj = Path(conf_path)
        if not conf_path_obj.is_dir():
            return []

        files: set[Path] = set()
        for pattern in patterns:
            for match in conf_path_obj.glob(pattern):
                if match.is_file() and self._is_valid_config_file(match):
                    files.add(match)
        return sorted(files)

    def _get_remote_config_files(
        self, conf_path: str, patterns: list[str]
    ) -> list[Path]:
        """Return the list of config files under a remote ``conf_path``."""
        files: set[Path] = set()
        for pattern in patterns:
            glob_path = f"{conf_path}/{pattern}"
            for match in self._fs.glob(glob_path):
                if self._fs.isfile(match) and self._is_valid_config_file(Path(match)):
                    files.add(Path(match))
        return sorted(files)

    def _is_valid_config_file(self, filepath: Path) -> bool:
        """Check whether a file should be loaded as a config file."""
        if self.ignore_hidden and any(
            part.startswith(".") for part in filepath.parts
        ):
            return False
        return filepath.suffix in {".yml", ".yaml", ".json"}

    def _load_config_file(
        self, filepath: Path, read_environment_variables: bool
    ) -> DictConfig:
        """Load a single config file into a ``DictConfig``."""
        with self._fs.open(str(filepath), "r") as f:
            content = f.read()

        if read_environment_variables:
            content = self._resolve_environment_variables(content)

        try:
            return OmegaConf.load(io.StringIO(content))
        except (ParserError, ScannerError) as exc:
            raise ValueError(
                f"Invalid YAML or JSON file {filepath}, unable to read line "
                f"{exc.problem_mark.line + 1}, position "
                f"{exc.problem_mark.column + 1}."
            ) from exc

    def _resolve_environment_variables(self, content: str) -> str:
        """Resolve environment variables in a raw config string."""
        return re.sub(
            r"\$\{oc\.env:([^}]+)\}",
            lambda m: self._resolve_env_var(m.group(1)),
            content,
        )

    def _resolve_env_var(self, expression: str) -> str:
        """Resolve a single ``oc.env`` expression to a YAML-safe string."""
        expression = expression.strip()
        parts = expression.split(",")
        var_name = parts[0].strip().strip("'\"")
        default = None
        if len(parts) > 1:
            default = parts[1].strip().strip("'\"")
        value = os.environ.get(var_name, default)
        if value is None:
            raise MissingConfigException(
                f"Failed to resolve environment variable '{var_name}'"
            )
        return json.dumps(value)

    def _resolve_from_raw_config(
        self,
        key: str,
        base_configs: dict[Path, DictConfig],
        env_configs: dict[Path, DictConfig],
        base_path: str,
        env_path: str,
        processed_files: set[Path],
    ) -> dict[str, Any]:
        """Merge and resolve raw per-file configs for ``key``."""
        base_merged = self._merge_configs(base_configs, key, base_path)
        env_merged = self._merge_configs(env_configs, key, env_path)

        if key == "parameters":
            merged = self._merge_parameters(base_merged, env_merged)
        else:
            merged = self._merge_configs(
                {**base_configs, **env_configs}, key, base_path
            )

        return self._resolve_config(merged, key)

    def _merge_configs(
        self, configs: dict[Path, DictConfig], key: str, conf_path: str
    ) -> DictConfig:
        """Merge a set of per-file configs, raising on duplicate top-level keys."""
        merged: DictConfig = OmegaConf.create({})
        for filepath in sorted(configs):
            config = configs[filepath]
            duplicates = set(merged.keys()) & set(config.keys())
            if duplicates:
                raise ValueError(
                    f"Duplicate keys found in {conf_path}: {', '.join(sorted(duplicates))}"
                )
            merged = OmegaConf.merge(merged, config)
        return merged

    def _merge_parameters(
        self, base_merged: DictConfig, env_merged: DictConfig
    ) -> DictConfig:
        """Merge parameters configs, allowing nested overrides."""
        return OmegaConf.merge(base_merged, env_merged)

    def _resolve_config(self, config: DictConfig, key: str) -> dict[str, Any]:
        """Resolve interpolations in a merged config and return a plain dict."""
        try:
            resolved = OmegaConf.to_container(config, resolve=True)
        except InterpolationResolutionError as exc:
            raise InterpolationResolutionError(
                f"Failed to resolve interpolation in {key} config: {exc}"
            ) from exc
        return resolved  # type: ignore[no-any-return]

    def _register_new_resolvers(
        self, custom_resolvers: dict[str, Callable | tuple[Callable, dict[str, Any]]]
    ) -> None:
        """Register custom resolvers with OmegaConf.

        A resolver can be given as a plain callable or as a ``(callable, kwargs)``
        tuple. The ``kwargs`` are passed on to ``OmegaConf.register_new_resolver()``.

        A resolver whose name is already registered is skipped, unless its ``kwargs``
        contain ``replace=True``, in which case it is re-registered so the existing
        resolver is replaced.
        """
        for name, resolver in custom_resolvers.items():
            if isinstance(resolver, tuple):
                callable_, kwargs = resolver
                kwargs = dict(kwargs)
                replace = bool(kwargs.pop("replace", False))
            else:
                callable_ = resolver
                kwargs = {}
                replace = False

            if OmegaConf.has_resolver(name) and not replace:
                _config_logger.debug(
                    "Custom resolver '%s' is already registered; skipping.", name
                )
                continue

            OmegaConf.register_new_resolver(
                name, callable_, replace=replace, **kwargs
            )

    def _register_globals_resolver(self) -> None:
        """Register the ``globals`` resolver used for globals interpolation."""
        if not OmegaConf.has_resolver("globals"):
            OmegaConf.register_new_resolver(
                "globals",
                lambda key: self._get_globals_value(key),
                use_cache=True,
            )

    def _get_globals_value(self, key: str) -> Any:
        """Return a value from the cached globals config."""
        if not hasattr(self, "_globals"):
            return None
        value = self._globals
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    def _register_runtime_params_resolver(self) -> None:
        """Register the ``runtime_params`` resolver."""
        if not OmegaConf.has_resolver("runtime_params"):
            OmegaConf.register_new_resolver(
                "runtime_params",
                self._resolve_runtime_params,
                use_cache=True,
            )

    def _resolve_runtime_params(self, key: str, default: Any = _NO_VALUE) -> Any:
        """Resolve a ``runtime_params`` interpolation."""
        if self._runtime_params_oc is None:
            self._runtime_params_oc = OmegaConf.create(self.runtime_params or {})

        value = self._runtime_params_oc
        for part in key.split("."):
            if not OmegaConf.is_dict(value) or part not in value:
                if default is not _NO_VALUE:
                    return default
                raise InterpolationResolutionError(
                    f"runtime_params key '{key}' not found"
                )
            value = value[part]

        self._runtime_params_hits.add(key)
        return value

    def _initialise_filesystem_and_protocol(
        self, conf_source: str
    ) -> tuple[Any, str]:
        """Initialise the filesystem and protocol for ``conf_source``."""
        protocol = _parse_filepath(conf_source)["protocol"]
        fs = fsspec.filesystem(protocol)
        return fs, protocol

    def load_and_merge_dir_config(
        self,
        conf_path: str,
        patterns: list[str],
        key: str,
        processed_files: set[Path],
        read_environment_variables: bool = False,
    ) -> dict[str, Any]:
        """Load and merge config files from a single directory.

        Kept as public API for backwards compatibility.
        """
        configs = self._read_dir_configs(
            conf_path, patterns, key, processed_files, read_environment_variables
        )
        merged = self._merge_configs(configs, key, conf_path)
        return self._resolve_config(merged, key)

    @property
    def conf_paths(self) -> list[str]:
        """Return the list of config paths to load from."""
        paths = [self._get_conf_env_path(self.base_env)]
        run_env = self.env or self.default_run_env
        if run_env and run_env != self.base_env:
            paths.append(self._get_conf_env_path(run_env))
        return paths

    def keys(self) -> KeysView[str]:
        """Return the keys of the config patterns."""
        return self.config_patterns.keys()

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self.config_patterns)

    def __len__(self) -> int:
        return len(self.config_patterns)

    def __contains__(self, key: object) -> bool:
        return key in self.config_patterns

    def __repr__(self) -> str:
        return (
            f"OmegaConfigLoader(conf_source={self.conf_source!r}, "
            f"env={self.env!r}, base_env={self.base_env!r}, "
            f"default_run_env={self.default_run_env!r})"
        )
