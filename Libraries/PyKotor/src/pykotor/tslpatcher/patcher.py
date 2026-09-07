from __future__ import annotations

import os
import pathlib
import shutil
import tempfile

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from pykotor.common.stream import BinaryReader, BinaryWriter
from pykotor.extract.capsule import Capsule
from pykotor.extract.file import ResourceIdentifier
from pykotor.extract.installation import Installation
from pykotor.tools.encoding import decode_bytes_with_fallbacks
from pykotor.tools.misc import is_capsule_file, is_mod_file, is_rim_file
from pykotor.tools.module import rim_to_mod
from pykotor.tools.path import CaseAwarePath
from pykotor.tslpatcher.config import PatcherConfig
from pykotor.tslpatcher.logger import PatchLogger
from pykotor.tslpatcher.memory import PatcherMemory
from pykotor.tslpatcher.mods.install import InstallFile, create_backup
from pykotor.tslpatcher.mods.nss import ModificationsNSS, MutableString
from pykotor.tslpatcher.mods.template import OverrideType
from pykotor.tslpatcher.mods.tlk import MergeTLK
from utility.error_handling import universal_simplify_exception
from utility.logger_util import RobustRootLogger
from utility.system.path import PurePath

if TYPE_CHECKING:
    from threading import Event

    from typing_extensions import Literal

    from pykotor.common.misc import Game
    from pykotor.resource.type import SOURCE_TYPES
    from pykotor.tslpatcher.mods.template import PatcherModifications
    from pykotor.tslpatcher.mods.tlk import ModificationsTLK


@dataclass
class _PatchTarget:
    exists: bool
    capsule: Capsule | None
    staged_capsule_path: CaseAwarePath | None = None


class ModInstaller:
    def __init__(
        self,
        mod_path: os.PathLike | str,
        game_path: os.PathLike | str,
        changes_ini_path: os.PathLike | str,
        logger: PatchLogger | None = None,
    ):
        """Initialize a Patcher instance.

        Args:
        ----
            mod_path: {Path to the mod directory}
            game_path: {Path to the game directory}
            changes_ini_path: {Path to the changes ini file}
            logger: {Optional logger instance}.

        Returns:
        -------
            self: {Returns the Patcher instance}

        Processing Logic:
        ----------------
            - Initialize the logger if not already defined.
            - Initialize parameters passed for game, mod and changes ini paths
            - Handle legacy changes ini path syntax (changes_ini_path used to just be a filename)
            - Initialize other attributes.
        """
        self.game_path: CaseAwarePath = self._resolve_folder(game_path)
        self.mod_path: CaseAwarePath = self._resolve_folder(mod_path)
        self.changes_ini_path: CaseAwarePath = CaseAwarePath.pathify(changes_ini_path)
        self.tslpatchdata_path: CaseAwarePath | None = None
        self.patch_data_path: CaseAwarePath = self.mod_path
        self.log: PatchLogger = logger or PatchLogger()
        self.game: Game | None = Installation.determine_game(self.game_path)
        resolved_changes_ini = self._find_case_insensitive_file(self.changes_ini_path)
        if resolved_changes_ini is not None:
            self.changes_ini_path = resolved_changes_ini
        else:  # Handle legacy syntax
            self.changes_ini_path = self.mod_path / self.changes_ini_path.name
            resolved_changes_ini = self._find_case_insensitive_file(self.changes_ini_path)
            if resolved_changes_ini is None:
                tslpatchdata_folder = self._resolve_relative_folder_within(
                    self.mod_path,
                    "tslpatchdata",
                    "tslpatchdata folder",
                )
                self.changes_ini_path = tslpatchdata_folder / self.changes_ini_path.name
                resolved_changes_ini = self._find_case_insensitive_file(self.changes_ini_path)
            if resolved_changes_ini is None:
                import errno
                msg = "Could not find the changes ini file on disk."
                raise FileNotFoundError(errno.ENOENT, msg, str(self.changes_ini_path))
            self.changes_ini_path = resolved_changes_ini

        changes_ini_folder = self._resolve_folder(self.changes_ini_path.parent)
        if changes_ini_folder.name.casefold() == "tslpatchdata":
            self.tslpatchdata_path = changes_ini_folder
            self.patch_data_path = changes_ini_folder

        self._config: PatcherConfig | None = None
        self._backup: CaseAwarePath | None = None
        self._processed_backup_files: set = set()

    @staticmethod
    def _find_case_insensitive_child(
        folder: CaseAwarePath,
        name: str,
        *,
        directory: bool | None = None,
    ) -> CaseAwarePath | None:
        if not folder.safe_isdir():
            return None

        matches: list[CaseAwarePath] = []
        try:
            for child in folder.safe_iterdir():
                if child.name.casefold() != name.casefold():
                    continue
                if directory is True and not child.safe_isdir():
                    continue
                if directory is False and not child.safe_isfile():
                    continue
                if child.name == name:
                    return CaseAwarePath.pathify(child)
                matches.append(CaseAwarePath.pathify(child))
        except OSError:
            return None

        return min(matches, key=lambda path: path.name) if matches else None

    @classmethod
    def _resolve_folder(cls, folder: os.PathLike | str) -> CaseAwarePath:
        absolute_folder = CaseAwarePath.pathify(os.path.abspath(os.fspath(folder)))
        current = CaseAwarePath.pathify(absolute_folder.anchor)

        for part in absolute_folder.parts[1:]:
            if part == ".":
                continue
            if part == "..":
                current = current.parent
                continue

            existing = cls._find_case_insensitive_child(current, part, directory=True)
            current = existing if existing is not None else current / part

        return current

    @staticmethod
    def _relative_path_parts(
        path: os.PathLike | str,
        description: str,
        *,
        allow_empty: bool = True,
    ) -> tuple[str, ...]:
        raw_path = os.fspath(path)
        if "\0" in raw_path:
            raise ValueError(f"Invalid {description}: paths cannot contain null bytes.")

        parsed_path = pathlib.PureWindowsPath(raw_path)
        if parsed_path.is_absolute() or parsed_path.drive or parsed_path.root:
            raise ValueError(f"Invalid {description} '{raw_path}': absolute paths are not allowed.")

        parts = tuple(part for part in parsed_path.parts if part not in {"", "."})
        if any(part == ".." for part in parts):
            raise ValueError(f"Invalid {description} '{raw_path}': parent-directory traversal is not allowed.")

        invalid_characters = frozenset('<>:"|?*')
        reserved_names = {
            "aux",
            "clock$",
            "con",
            "nul",
            "prn",
            *(f"com{index}" for index in range(1, 10)),
            *(f"lpt{index}" for index in range(1, 10)),
        }
        for part in parts:
            if part.endswith((" ", ".")):
                raise ValueError(f"Invalid {description} '{raw_path}': path components cannot end with a space or period.")
            if any(character in invalid_characters or ord(character) < 32 for character in part):
                raise ValueError(f"Invalid {description} '{raw_path}': path contains characters invalid on Windows.")
            if part.split(".", 1)[0].casefold() in reserved_names:
                raise ValueError(f"Invalid {description} '{raw_path}': path uses a reserved Windows device name.")

        if not allow_empty and not parts:
            raise ValueError(f"Invalid {description}: a path value is required.")
        return parts

    @staticmethod
    def _ensure_within_root(
        path: os.PathLike | str,
        root: os.PathLike | str,
        description: str,
    ) -> CaseAwarePath:
        candidate = CaseAwarePath.pathify(path)
        root_real = os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(root))))
        candidate_real = os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(candidate))))
        try:
            common_path = os.path.commonpath((root_real, candidate_real))
        except ValueError as exc:
            raise ValueError(f"Invalid {description} '{candidate}': path is outside '{root}'.") from exc
        if common_path != root_real:
            raise ValueError(f"Invalid {description} '{candidate}': path is outside '{root}'.")
        return candidate

    @classmethod
    def _resolve_relative_folder_within(
        cls,
        root: os.PathLike | str,
        relative_path: os.PathLike | str,
        description: str,
    ) -> CaseAwarePath:
        root_path = CaseAwarePath.pathify(root)
        parts = cls._relative_path_parts(relative_path, description)
        resolved_path = cls._resolve_folder(root_path.joinpath(*parts))
        return cls._ensure_within_root(resolved_path, root_path, description)

    @classmethod
    def _resolve_relative_file_within(
        cls,
        root: os.PathLike | str,
        relative_path: os.PathLike | str,
        description: str,
    ) -> CaseAwarePath:
        root_path = CaseAwarePath.pathify(root)
        parts = cls._relative_path_parts(relative_path, description, allow_empty=False)
        requested_path = root_path.joinpath(*parts)
        resolved_parent = cls._resolve_folder(requested_path.parent)
        cls._ensure_within_root(resolved_parent, root_path, description)
        existing_path = cls._find_case_insensitive_child(resolved_parent, requested_path.name, directory=False)
        resolved_path = existing_path if existing_path is not None else resolved_parent / requested_path.name
        return cls._ensure_within_root(resolved_path, root_path, description)

    @classmethod
    def _resolve_file_path_within(
        cls,
        root: os.PathLike | str,
        filepath: os.PathLike | str,
        description: str,
    ) -> CaseAwarePath:
        root_path = CaseAwarePath.pathify(root)
        requested_path = CaseAwarePath.pathify(os.path.abspath(os.fspath(filepath)))
        cls._ensure_within_root(requested_path, root_path, description)
        resolved_parent = cls._resolve_folder(requested_path.parent)
        cls._ensure_within_root(resolved_parent, root_path, description)
        existing_path = cls._find_case_insensitive_child(resolved_parent, requested_path.name, directory=False)
        resolved_path = existing_path if existing_path is not None else resolved_parent / requested_path.name
        return cls._ensure_within_root(resolved_path, root_path, description)

    def _resolve_source_file_path(
        self,
        filepath: os.PathLike | str,
        description: str,
    ) -> CaseAwarePath:
        all_pooled_mod_locations = [self.mod_path, self.patch_data_path, self.tslpatchdata_path]
        
        #removingduplicates and None
        source_roots = list(dict.fromkeys(modfolder for modfolder in all_pooled_mod_locations if modfolder is not None))
        
        # Isolating the relevant filename
        filename = os.path.basename(os.fspath(filepath))

        for source_root in source_roots:
            mod_path = source_root / filename
            try:
                resolved = self._resolve_file_path_within(source_root, mod_path, description)
                if resolved.exists():
                    return resolved
            except ValueError:
                continue
                
        raise ValueError(f"Invalid {description} '{filepath}': path is outside the mod data folders.")

    @classmethod
    def _validate_output_filename(cls, filename: str) -> str:
        parts = cls._relative_path_parts(filename, "output filename", allow_empty=False)
        if len(parts) != 1:
            raise ValueError(f"Invalid output filename '{filename}': subdirectories are not allowed in !SaveAs/!Filename.")
        return parts[0]

    def _resolve_patch_output_paths(
        self,
        patch: PatcherModifications,
    ) -> tuple[CaseAwarePath, CaseAwarePath]:
        patch.saveas = self._validate_output_filename(patch.saveas).lower()
        destination_parts = self._relative_path_parts(patch.destination, "patch destination")
        requested_destination = self.game_path.joinpath(*destination_parts)

        if is_capsule_file(patch.destination):
            destination_folder = self._resolve_folder(requested_destination.parent)
            self._ensure_within_root(destination_folder, self.game_path, "patch destination")
            output_path = destination_folder / requested_destination.name
            container_path = output_path
        else:
            destination_folder = self._resolve_folder(requested_destination)
            self._ensure_within_root(destination_folder, self.game_path, "patch destination")
            output_path = destination_folder / patch.saveas
            container_path = destination_folder

        self._ensure_within_root(output_path, self.game_path, "patch output")
        existing_output = self._find_case_insensitive_file(output_path)
        if existing_output is not None:
            self._ensure_within_root(existing_output, self.game_path, "patch output")
        return container_path, output_path

    def _validate_patch_paths(self, patches: list[PatcherModifications]) -> None:
        for patch in patches:
            self._resolve_patch_output_paths(patch)

            source_folder = self._resolve_relative_folder_within(
                self.patch_data_path,
                patch.sourcefolder,
                "patch source folder",
            )
            self._resolve_relative_file_within(
                source_folder,
                patch.sourcefile,
                "patch source file",
            )

            compiler_path = getattr(patch, "nwnnsscomp_path", None)
            if compiler_path is not None:
                self._resolve_source_file_path(compiler_path, "script compiler")

            for modifier in getattr(patch, "modifiers", ()):
                tlk_filepath = getattr(modifier, "tlk_filepath", None)
                if tlk_filepath is not None:
                    self._resolve_source_file_path(tlk_filepath, "TLK source file")

    @classmethod
    def _find_case_insensitive_file(cls, filepath: os.PathLike | str) -> CaseAwarePath | None:
        requested_path = CaseAwarePath.pathify(filepath)
        parent = cls._resolve_folder(requested_path.parent)
        return cls._find_case_insensitive_child(parent, requested_path.name, directory=False)

    @classmethod
    def _lowercase_file_path(cls, filepath: os.PathLike | str) -> CaseAwarePath:
        requested_path = CaseAwarePath.pathify(filepath)
        parent = cls._resolve_folder(requested_path.parent)
        lowercase_path = parent / requested_path.name.lower()
        existing_path = cls._find_case_insensitive_child(parent, lowercase_path.name, directory=False)

        if existing_path is None or existing_path.name == lowercase_path.name:
            return lowercase_path

        temp_stem = f".{lowercase_path.name}.holopatcher"
        temp_path = parent / f"{temp_stem}.tmp"
        index = 2
        while cls._find_case_insensitive_child(parent, temp_path.name) is not None:
            temp_path = parent / f"{temp_stem}.{index}.tmp"
            index += 1

        os.replace(existing_path, temp_path)
        try:
            os.replace(temp_path, lowercase_path)
        except Exception:
            os.replace(temp_path, existing_path)
            raise
        return lowercase_path

    def _prepare_output_path(self, patch: PatcherModifications) -> CaseAwarePath:
        container_path, output_path = self._resolve_patch_output_paths(patch)
        lowercase_output_path = self._lowercase_file_path(output_path)
        self._ensure_within_root(lowercase_output_path, self.game_path, "patch output")
        return lowercase_output_path if is_capsule_file(patch.destination) else container_path

    def _skip_protected_install(self, patch: PatcherModifications) -> bool:
        if (
            not isinstance(patch, InstallFile)
            or not patch.is_protected_replacement()
            or is_capsule_file(patch.destination)
        ):
            return False

        _, output_path = self._resolve_patch_output_paths(patch)
        existing_output = self._find_case_insensitive_file(output_path)
        if existing_output is None:
            return False

        local_folder = self.game_path.name if patch.destination.strip("/\\") == "." else patch.destination
        self.log.add_warning(
            f"Skipping protected existing file '{patch.saveas}' in the '{local_folder}' folder; "
            "InstallList cannot replace EXE, TLK, KEY, or BIF files.",
        )
        return True

    def config(self) -> PatcherConfig:
        """Returns the PatcherConfig object associated with the mod installer.

        The object is created when the method is first called then cached for future calls.
        """
        if self._config is not None:
            return self._config

        ini_file_bytes: bytes = BinaryReader.load_file(self.changes_ini_path)
        ini_text: str
        try:
            ini_text = decode_bytes_with_fallbacks(ini_file_bytes)
        except UnicodeDecodeError:
            self.log.add_warning(f"Could not determine encoding of '{self.changes_ini_path.name}'. Attempting to force load...")
            ini_text = ini_file_bytes.decode(errors="ignore")

        self._config = PatcherConfig()
        self._config.load(ini_text, self.mod_path, self.log, self.tslpatchdata_path)

        if self._config.required_files:
            override_folder = self._resolve_relative_folder_within(
                self.game_path,
                "override",
                "required-file destination",
            )
            for i, files in enumerate(self._config.required_files):
                for file in files:
                    requiredfile_path = self._resolve_relative_file_within(
                        override_folder,
                        file,
                        "required file",
                    )
                    if not requiredfile_path.safe_isfile():
                        requiredfile_path = None
                    if requiredfile_path is None:
                        required_message = self._config.required_messages[i].strip()
                        if not required_message:
                            required_message = f"Cannot locate required file {file}, unable to continue with install!"
                        raise ImportError(required_message)
        return self._config

    def backup(self) -> tuple[CaseAwarePath, set]:
        """Creates a backup of the patch files.

        Args:
        ----
            self: The Patcher object

        Returns:
        -------
            tuple[CaseAwarePath, set]: Returns a tuple containing the backup directory path and a set of processed backup files

        Processing Logic:
        ----------------
            - Checks if a backup folder was already initialized and return that and the currently processed files if so
            - Finds the mod path directory to backup from
            - Generates a timestamped subdirectory name
            - Removes any existing uninstall directories
            - Creates the backup directory
            - Returns the backup directory and new hashset that'll contain the processed files
        """
        if self._backup:
            return (self._backup, self._processed_backup_files)
        package_root = self.mod_path
        timestamp: str = datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d_%H.%M.%S")
        if self.mod_path.name.casefold() == "tslpatchdata":
            package_root = self.mod_path.parent
        elif self._resolve_relative_folder_within(
            self.mod_path,
            "tslpatchdata",
            "tslpatchdata folder",
        ).safe_isdir():
            package_root = self.mod_path
        uninstall_dir = self._resolve_relative_folder_within(
            package_root,
            "uninstall",
            "uninstall folder",
        )
        try:  # sourcery skip: remove-redundant-exception
            if uninstall_dir.is_dir():
                shutil.rmtree(uninstall_dir)
        except (PermissionError, OSError) as e:
            self.log.add_warning(f"Could not initialize uninstall directory: {universal_simplify_exception(e)}")
        backup_parent = self._resolve_relative_folder_within(
            package_root,
            "backup",
            "backup folder",
        )
        backup_dir = self._ensure_within_root(
            backup_parent / timestamp,
            package_root,
            "backup folder",
        )
        try:  # sourcery skip: remove-redundant-exception
            backup_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            self.log.add_error(f"Could not create backup folder: {universal_simplify_exception(e)}")
            raise
        self.log.add_note(f"Using backup directory: '{backup_dir}'")
        self._backup = backup_dir
        self._processed_backup_files = set()
        return (self._backup, self._processed_backup_files)

    def handle_capsule_and_backup(
        self,
        patch: PatcherModifications,
        output_container_path: CaseAwarePath,
    ) -> _PatchTarget:
        """Prepare the output target and back up any pre-existing destination."""
        if is_capsule_file(patch.destination):
            module_root = Installation.get_module_root(output_container_path)
            tslrcm_omitted_rims = ("702KOR", "401DXN")
            if module_root.upper() not in tslrcm_omitted_rims and is_rim_file(output_container_path):
                self.log.add_warning(
                    f"This mod is patching RIM file Modules/{output_container_path.name}!\n"
                    "Patching RIMs is highly incompatible, not recommended, and widely considered bad practice. "
                    "Please request the mod developer to fix this.",
                )

            staged_capsule_path: CaseAwarePath | None = None
            capsule_path = output_container_path
            if not output_container_path.safe_isfile():
                if not is_mod_file(output_container_path):
                    import errno

                    msg = (
                        f"The capsule '{patch.destination}' did not exist, or permission issues occurred, when "
                        f"attempting to {patch.action.lower().rstrip()} '{patch.sourcefile}'. Skipping file..."
                    )
                    raise FileNotFoundError(errno.ENOENT, msg, str(output_container_path))

                self.log.add_note(
                    f"IMPORTANT! The module at path '{output_container_path}' did not exist, staging one from:"
                    f"\n    Modules/{module_root}.rim"
                    f"\n    Modules/{module_root}_s.rim"
                    + (f"\n    Modules/{module_root}_dlg.erf" if self.game is not None and self.game.is_k2() else ""),
                )
                modules_folder = self._resolve_relative_folder_within(
                    self.game_path,
                    "modules",
                    "module folder",
                )
                module_source_names = [
                    f"{module_root}.rim",
                    f"{module_root}_s.rim",
                ]
                if self.game is not None and self.game.is_k2():
                    module_source_names.append(f"{module_root}_dlg.erf")
                for module_source_name in module_source_names:
                    module_source = self._resolve_relative_file_within(
                        modules_folder,
                        module_source_name,
                        "module source",
                    )
                    if module_source.safe_exists():
                        self._ensure_within_root(module_source, modules_folder, "module source")

                output_container_path.parent.mkdir(parents=True, exist_ok=True)
                file_descriptor, staged_name = tempfile.mkstemp(
                    prefix=".holopatcher_",
                    suffix=".mod",
                    dir=output_container_path.parent,
                )
                os.close(file_descriptor)
                staged_capsule_path = CaseAwarePath.pathify(staged_name)
                try:
                    rim_to_mod(staged_capsule_path, modules_folder, module_root, self.game)
                except Exception as exc:
                    staged_capsule_path.unlink(missing_ok=True)
                    msg = f"Failed to build module '{output_container_path.name}': {exc}"
                    self.log.add_error(msg)
                    raise
                capsule_path = staged_capsule_path
            else:
                backup_subdirectory = PurePath(os.path.relpath(output_container_path.parent, self.game_path))
                create_backup(self.log, output_container_path, *self.backup(), backup_subdirectory)

            capsule = Capsule(capsule_path)
            exists = capsule.contains(*ResourceIdentifier.from_path(patch.saveas).unpack())
            return _PatchTarget(exists, capsule, staged_capsule_path)

        backup_subdirectory = PurePath(os.path.relpath(output_container_path, self.game_path))
        create_backup(
            self.log,
            output_container_path / patch.saveas,
            *self.backup(),
            backup_subdirectory,
        )
        exists = output_container_path.joinpath(patch.saveas).is_file()
        return _PatchTarget(exists, None)

    def _commit_staged_capsule(
        self,
        staged_capsule_path: CaseAwarePath,
        output_container_path: CaseAwarePath,
    ) -> None:
        """Atomically install a newly constructed capsule and register it for uninstall."""
        backup_subdirectory = PurePath(os.path.relpath(output_container_path.parent, self.game_path))
        create_backup(
            self.log,
            output_container_path,
            *self.backup(),
            backup_subdirectory,
            is_new_file=not output_container_path.safe_isfile(),
        )
        os.replace(staged_capsule_path, output_container_path)

    def load_resource_file(self, source: SOURCE_TYPES) -> bytes:
        # if self._config and self._config.ignore_file_extensions:
        #    return read_resource(source)
        with BinaryReader.from_auto(source) as reader:
            return reader.read_all()

    def lookup_resource(
        self,
        patch: PatcherModifications,
        output_container_path: CaseAwarePath,
        exists_at_output_location: bool | None = None,  # noqa: FBT001
        capsule: Capsule | None = None,
    ) -> bytes | None:
        """Looks up the file/resource that is expected to be patched.

        Args:
        ----
            patch: PatcherModifications - The desired patch information.
            output_container_path: CaseAwarePath - Path to output container (capsule/folder)
            exists_at_output_location: bool | None - Whether resource exists at destination location
            capsule: Capsule | None - Capsule to be patched, if one

        Returns:
        -------
            bytes | None - Loaded resource bytes or None

        Processing Logic:
        ----------------
            - Check if file should be replaced or doesn't exist at output, load from mod path
            - Otherwise, load the file to be patched from the destination if it exists.
                - If no capsule, it's a file and load it directly as a file.
                - If destination is a capsule, pull the resource from the capsule.
            - Return None and log error on failure (IO exceptions, permission issues, etc)
        """
        try:
            if patch.replace_file or not exists_at_output_location:
                source_folder = self._resolve_relative_folder_within(
                    self.patch_data_path,
                    patch.sourcefolder,
                    "patch source folder",
                )
                source_path = self._resolve_relative_file_within(
                    source_folder,
                    patch.sourcefile,
                    "patch source file",
                )
                self._ensure_within_root(source_path, self.patch_data_path, "patch source file")
                return self.load_resource_file(source_path)
            if capsule is None:
                return self.load_resource_file(output_container_path / patch.saveas)
            return capsule.resource(*ResourceIdentifier.from_path(patch.saveas).unpack())
        except OSError as e:
            self.log.add_error(f"Could not load source file to {patch.action.lower().strip()}:{os.linesep}{universal_simplify_exception(e)}")
            return None

    def handle_modrim_shadow(
        self,
        patch: PatcherModifications,
        output_container_path: CaseAwarePath,
    ):
        """Check if a patch is being installed into a rim and overshadowed by a .mod."""
        # uncomment and define the attrs if we decide this should be configurable.
        # modrim_type: str = patch.modrim_type.lower().strip()
        # if not modrim_type or modrim_type == ignore
        #    return
        mod_path = output_container_path.with_name(
            f"{Installation.get_module_root(output_container_path.name)}.mod".lower(),
        )
        existing_mod_path = self._find_case_insensitive_file(mod_path)
        if output_container_path != mod_path and existing_mod_path is not None:
            self.log.add_warning(
                f"This mod intends to install '{patch.saveas}' into '{patch.destination}', "
                f"but is overshadowed by the existing '{mod_path.name}'!",
            )

    def handle_override_type(self, patch: PatcherModifications):
        """Handles the desired behavior set by the !OverrideType tslpatcher var for the specified patch.

        Args:
        ----
            patch: PatcherModifications - The patch modification object.

        Processes the override type:
            - Checks if override type is empty or set to ignore and returns early.
            - Gets the override resource path.
            - If the path exists:
                - For rename, renames the file with incrementing number if filename exists.
                - For warn, logs a warning that the file is shadowing the mod's changes.
        """
        override_type: str = patch.override_type.lower().strip()
        if not override_type or override_type == OverrideType.IGNORE:
            return

        override_dir = self._resolve_relative_folder_within(
            self.game_path,
            "override",
            "Override folder",
        )
        override_resource_path = self._find_case_insensitive_file(override_dir / patch.saveas)
        if override_resource_path is not None:
            if override_type == OverrideType.RENAME:
                renamed_file_path: CaseAwarePath = override_dir / f"old_{patch.saveas}".lower()
                i = 2
                filestem: str = renamed_file_path.stem
                while self._find_case_insensitive_file(renamed_file_path) is not None:
                    renamed_file_path = renamed_file_path.parent / f"{filestem} ({i}){renamed_file_path.suffix}".lower()
                    i += 1
                try:
                    shutil.move(str(override_resource_path), str(renamed_file_path))
                except Exception as e:  # pylint: disable=W0718  # noqa: BLE001
                    # Handle exceptions such as permission errors or file in use.
                    self.log.add_error(f"Could not rename '{patch.saveas}' to '{renamed_file_path.name}' in the Override folder: {universal_simplify_exception(e)}")  # noqa: E501
            elif override_type == OverrideType.WARN:
                self.log.add_warning(f"A resource located at '{override_resource_path}' is shadowing this mod's changes in {patch.destination}!")  # noqa: E501

    def should_patch(
        self,
        patch: PatcherModifications,
        exists: bool | None = False,  # noqa: FBT002, FBT001
        capsule: Capsule | None = None,
    ) -> bool:
        """Log information about the patch, including source and destination.

        The name of this function can be misleading, it only returns False if the capsule was not found (error)
        or an InstallList patch already exists at the output location without the Replace#= prefix. Otherwise, it is
        mostly used for logging purposes.

        Args:
        ----
            patch (PatcherModifications): - The patch details
            exists (bool | None): - Whether the target file already exists
            capsule (Capsule | None): - The target capsule if patching one

        Returns:
        -------
            bool - Whether the patch should be applied
                False if the capsule was not found (error)
                False if an InstallList patch already exists at destination and patch configured to replace existing file or not (!ReplaceFile/#Replace=filename)
                True otherwise.

        Processing Logic:
        ----------------
            - Determines the local folder and container type from the patch details
            - Checks if the patch replaces an existing file and logs the action
            - Checks if the file already exists and the patch settings allow skipping
            - Checks if the target capsule exists if patching one
            - Logs the patching action
            - Returns True if the patch should be applied.
        """  # noqa: E501
        local_folder: str = self.game_path.name if patch.destination.strip("\\").strip("/") == "." else patch.destination
        container_type: Literal["folder", "archive"] = "folder" if capsule is None else "archive"

        if (
            isinstance(patch, InstallFile)
            and capsule is None
            and exists
            and patch.is_protected_replacement()
        ):
            self.log.add_warning(
                f"Skipping protected existing file '{patch.saveas}' in the '{local_folder}' folder; "
                "InstallList cannot replace EXE, TLK, KEY, or BIF files.",
            )
            return False

        if patch.replace_file and exists:
            saveas_str: str = f"'{patch.saveas}' in" if patch.saveas.casefold() != patch.sourcefile.casefold() else "in"
            self.log.add_note(f"{patch.action[:-1]}ing '{patch.sourcefile}' and replacing existing file {saveas_str} the '{local_folder}' {container_type}")  # noqa: E501
            return True

        if not patch.skip_if_not_replace and not patch.replace_file and exists:
            self.log.add_note(f"{patch.action[:-1]}ing existing file '{patch.saveas}' in the '{local_folder}' {container_type}")
            return True

        if patch.skip_if_not_replace and not patch.replace_file and exists:  # [InstallList] only
            self.log.add_note(f"'{patch.saveas}' already exists in the '{local_folder}' {container_type}. Skipping file...")
            return False

        if capsule is not None and not capsule.filepath().safe_isfile():
            self.log.add_error(f"The capsule '{patch.destination}' did not exist when attempting to {patch.action.lower().rstrip()} '{patch.sourcefile}'. Skipping file...")  # noqa: E501
            return False

        save_type: str = "adding" if capsule is not None and patch.saveas.casefold() == patch.sourcefile.casefold() else "saving"
        saving_as_str = f"as '{patch.saveas}' in" if patch.saveas.casefold() != patch.sourcefile.casefold() else "to"
        self.log.add_note(f"{patch.action[:-1]}ing '{patch.sourcefile}' and {save_type} {saving_as_str} the '{local_folder}' {container_type}")
        return True

    def install(
        self,
        should_cancel: Event | None = None,
        progress_update_func: Callable | None = None,
    ):  # noqa: C901
        """Install every configured patch and report its actual outcome."""
        if self.game is None:
            msg = "Chosen KOTOR directory is not a valid installation - cannot initialize ModInstaller."
            raise RuntimeError(msg)

        memory = PatcherMemory()
        config: PatcherConfig = self.config()
        configured_patches: list[PatcherModifications] = [
            config.patches_tlk,
            *config.install_list,
            *config.patches_2da,
            *config.patches_gff,
            *config.patches_ncs,
            *config.patches_nss,
            *config.patches_ssf,
        ]
        self._validate_patch_paths(configured_patches)
        self._add_compilelist_dependencies(config)
        patches_list: list[PatcherModifications] = [
            *self.get_tlk_patches(config),
            *config.install_list,
            *config.patches_2da,
            *config.patches_gff,
            *config.patches_ncs,
            *config.patches_nss,
            *config.patches_ssf,
        ]
        self._validate_patch_paths(patches_list)
        self.log.reset_patch_counts(len(patches_list))
        installation_errors_before = len(self.log.errors)
        installation_warnings_before = len(self.log.warnings)

        finished_preprocessed_scripts = False
        compile_workspace: tempfile.TemporaryDirectory | None = None
        compile_workspace_path: CaseAwarePath | None = None
        cancelled = False

        try:
            for patch_index, patch in enumerate(patches_list):
                if should_cancel is not None and should_cancel.is_set():
                    remaining = len(patches_list) - patch_index
                    self.log.skip_patch(remaining)
                    self.log.add_warning(f"Installation cancelled with {remaining} operations remaining.")
                    if progress_update_func is not None:
                        for _ in range(remaining):
                            progress_update_func()
                    cancelled = True
                    break

                outcome = "failed"
                target: _PatchTarget | None = None
                errors_before = len(self.log.errors)
                try:
                    if self._skip_protected_install(patch):
                        outcome = "skipped"
                        continue

                    # CompileList sources and includes must be preprocessed after
                    # all token-producing patches have run. The workspace lives
                    # outside the mod package and is removed in the outer finally.
                    if not finished_preprocessed_scripts and isinstance(patch, ModificationsNSS):
                        if compile_workspace is None:
                            compile_workspace = tempfile.TemporaryDirectory(prefix="holopatcher_nss_")
                            compile_workspace_path = CaseAwarePath.pathify(compile_workspace.name)
                        self._prepare_compilelist(
                            config,
                            self.log,
                            memory,
                            self.game,
                            compile_workspace_path,
                        )
                        finished_preprocessed_scripts = True

                    output_container_path = self._prepare_output_path(patch)
                    target = self.handle_capsule_and_backup(patch, output_container_path)
                    if not self.should_patch(patch, target.exists, target.capsule):
                        outcome = "skipped"
                        continue

                    data_to_patch = self.lookup_resource(
                        patch,
                        output_container_path,
                        target.exists,
                        target.capsule,
                    )
                    if data_to_patch is None:
                        self.log.add_error(
                            f"Could not locate resource to {patch.action.lower().strip()}: '{patch.sourcefile}'",
                        )
                        continue
                    if not data_to_patch:
                        self.log.add_note(f"'{patch.sourcefile}' has no content/data and is completely empty.")

                    patched_data: bytes | Literal[True] = patch.patch_resource(
                        data_to_patch,
                        memory,
                        self.log,
                        self.game,
                    )
                    if patched_data is True:
                        self.log.add_note(
                            f"Skipping '{patch.sourcefile}' - patch_resource determined that this file can be skipped.",
                        )
                        outcome = "skipped"
                        continue

                    if target.capsule is not None:
                        self.handle_override_type(patch)
                        self.handle_modrim_shadow(patch, output_container_path)
                        target.capsule.add(*ResourceIdentifier.from_path(patch.saveas).unpack(), patched_data)
                        if target.staged_capsule_path is not None:
                            self._commit_staged_capsule(target.staged_capsule_path, output_container_path)
                    else:
                        output_container_path.mkdir(exist_ok=True, parents=True)
                        BinaryWriter.dump(output_container_path / patch.saveas, patched_data)

                    outcome = "failed" if len(self.log.errors) > errors_before else "completed"
                except Exception as exc:  # pylint: disable=W0718  # noqa: BLE001
                    exc_type, exc_msg = universal_simplify_exception(exc)
                    msg = f"An error occurred in patchlist {patch.__class__.__name__}:\n{exc_type}: {exc_msg}\n"
                    self.log.add_error(msg)
                    RobustRootLogger().exception(msg)
                    outcome = "failed"
                finally:
                    if (
                        target is not None
                        and target.staged_capsule_path is not None
                        and target.staged_capsule_path.safe_exists()
                    ):
                        try:
                            target.staged_capsule_path.unlink()
                        except OSError as exc:
                            self.log.add_warning(
                                f"Could not remove staged module '{target.staged_capsule_path}': "
                                f"{universal_simplify_exception(exc)}",
                            )

                    if outcome == "completed":
                        self.log.complete_patch()
                    elif outcome == "skipped":
                        self.log.skip_patch()
                    else:
                        self.log.fail_patch()

                    if progress_update_func is not None:
                        progress_update_func()
        finally:
            if compile_workspace is not None:
                try:
                    if (
                        config.save_processed_scripts != 0
                        and finished_preprocessed_scripts
                        and compile_workspace_path is not None
                    ):
                        self._save_processed_scripts(compile_workspace_path)
                except Exception as exc:  # noqa: BLE001
                    self.log.add_error(
                        f"Could not save processed CompileList scripts: {universal_simplify_exception(exc)}",
                    )
                finally:
                    compile_workspace.cleanup()

        warning_count = len(self.log.warnings) - installation_warnings_before
        error_count = len(self.log.errors) - installation_errors_before
        operation_label = "operation" if self.log.patches_configured == 1 else "operations"
        warning_label = "warning" if warning_count == 1 else "warnings"
        error_label = "error" if error_count == 1 else "errors"
        summary = (
            f"Processed {self.log.patches_configured} {operation_label}: "
            f"{self.log.patches_completed} completed, "
            f"{self.log.patches_skipped} skipped, "
            f"{self.log.patches_failed} failed, "
            f"{warning_count} {warning_label}, "
            f"{error_count} {error_label}."
        )
        if cancelled:
            self.log.add_warning(f"Installation cancelled. {summary}")
        elif self.log.patches_failed or error_count:
            self.log.add_error(f"Installation completed with errors. {summary}")
        else:
            self.log.add_note(f"Successfully completed installation. {summary}")

    def _save_processed_scripts(self, workspace: CaseAwarePath) -> CaseAwarePath:
        package_root = self.mod_path.parent if self.mod_path.name.casefold() == "tslpatchdata" else self.mod_path
        output_parent = self._resolve_relative_folder_within(
            package_root,
            "processed_nss",
            "processed CompileList output folder",
        )
        output_parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d_%H.%M.%S.%f")
        output_folder = self._ensure_within_root(
            output_parent / timestamp,
            package_root,
            "processed CompileList output folder",
        )
        shutil.copytree(workspace, output_folder)
        self.log.add_note(f"Saved processed CompileList scripts to '{output_folder}'.")
        return output_folder

    def _add_compilelist_dependencies(self, config: PatcherConfig):
        if not config.patches_nss:
            return

        existing_install = next(
            (
                patch
                for patch in config.install_list
                if patch.saveas.casefold() == "nwscript.nss"
                and patch.destination.strip("/\\").casefold() == "override"
            ),
            None,
        )
        if existing_install is not None:
            return

        source_folders = [".", *(patch.sourcefolder for patch in config.patches_nss)]
        checked_folders: set[str] = set()
        for source_folder in source_folders:
            source_path = self._resolve_relative_folder_within(
                self.patch_data_path,
                source_folder,
                "CompileList source folder",
            )
            normalized_source_path = str(source_path).casefold()
            if normalized_source_path in checked_folders:
                continue
            checked_folders.add(normalized_source_path)

            nwscript_path = self._resolve_relative_file_within(
                source_path,
                "nwscript.nss",
                "CompileList dependency",
            )
            if not nwscript_path.safe_isfile():
                continue

            install = InstallFile("nwscript.nss", replace_existing=True)
            install.sourcefolder = source_folder
            config.install_list.append(install)
            return

    def _prepare_compilelist(
        self,
        config: PatcherConfig,
        log: PatchLogger,
        memory: PatcherMemory,
        game: Game,
        temp_script_folder: CaseAwarePath,
    ) -> CaseAwarePath | None:
        """tslpatchdata should be read-only, this allows us to replace memory tokens while ensuring include scripts work correctly."""  # noqa: D403, E501
        if not config.patches_nss:
            return None

        # Copy NSS sources and includes to a system temporary directory where
        # tokens can be replaced without modifying the mod package.
        if temp_script_folder.safe_isdir():
            shutil.rmtree(temp_script_folder, ignore_errors=True)
        temp_script_folder.mkdir(exist_ok=True, parents=True)

        include_folders: list[CaseAwarePath] = []
        seen_include_folders: set[str] = set()
        for source_folder in [".", *(patch.sourcefolder for patch in config.patches_nss)]:
            source_path = self._resolve_relative_folder_within(
                self.patch_data_path,
                source_folder,
                "CompileList source folder",
            )
            normalized_source_path = str(source_path).casefold()
            if normalized_source_path in seen_include_folders:
                continue
            seen_include_folders.add(normalized_source_path)
            include_folders.append(source_path)

        patch_source_folders: list[CaseAwarePath] = []
        seen_patch_source_folders: set[str] = set()
        for patch in config.patches_nss:
            source_path = self._resolve_relative_folder_within(
                self.patch_data_path,
                patch.sourcefolder,
                "CompileList source folder",
            )
            normalized_source_path = str(source_path).casefold()
            if normalized_source_path in seen_patch_source_folders:
                continue
            seen_patch_source_folders.add(normalized_source_path)
            patch_source_folders.append(source_path)

        working_folders: dict[str, CaseAwarePath] = {}
        script_count = 0
        for index, source_path in enumerate(patch_source_folders):
            working_folder = temp_script_folder / f"source_{index}"
            working_folder.mkdir(exist_ok=True, parents=True)

            # Make includes from every CompileList source directory available,
            # while allowing the current source directory to take precedence.
            normalized_source_path = str(source_path).casefold()
            ordered_sources = [
                path
                for path in include_folders
                if str(path).casefold() != normalized_source_path
            ]
            ordered_sources.append(source_path)
            for include_source in ordered_sources:
                if not include_source.safe_isdir():
                    continue
                for source_file in sorted(include_source.safe_iterdir(), key=lambda path: path.name.casefold()):
                    if source_file.suffix.lower() != ".nss" or not source_file.safe_isfile():
                        continue
                    safe_source_file = self._resolve_file_path_within(
                        self.patch_data_path,
                        source_file,
                        "CompileList source file",
                    )
                    shutil.copy2(safe_source_file, working_folder / source_file.name.lower())

            scripts = [
                script
                for script in sorted(working_folder.safe_iterdir(), key=lambda path: path.name.casefold())
                if script.suffix.lower() == ".nss" and script.safe_isfile()
            ]
            script_count += len(scripts)
            for script in scripts:
                log.add_verbose(f"Parsing tokens in '{script.name}'...")
                with script.open(mode="rb") as file:
                    content = MutableString(decode_bytes_with_fallbacks(file.read()))
                ModificationsNSS(script.name).apply(content, memory, log, game)
                with script.open(mode="w", encoding="windows-1252") as file:
                    file.write(content.value)

            working_folders[str(source_path).casefold()] = working_folder

        log.add_verbose(f"Preprocessed #StrRef# and #2DAMEMORY# tokens in {script_count} CompileList source and include files.")
        for nss_patch in config.patches_nss:
            source_path = self._resolve_relative_folder_within(
                self.patch_data_path,
                nss_patch.sourcefolder,
                "CompileList source folder",
            )
            nss_patch.temp_script_folder = working_folders[str(source_path).casefold()]
            if nss_patch.nwnnsscomp_path is not None:
                nss_patch.nwnnsscomp_path = self._resolve_source_file_path(
                    nss_patch.nwnnsscomp_path,
                    "script compiler",
                )
        return temp_script_folder

    def get_tlk_patches(self, config: PatcherConfig) -> list[ModificationsTLK]:
        tlk_patches: list[ModificationsTLK] = []
        patches_tlk: ModificationsTLK = config.patches_tlk

        if not patches_tlk.modifiers:
            return tlk_patches

        for modifier in patches_tlk.modifiers:
            tlk_filepath = getattr(modifier, "tlk_filepath", None)
            if tlk_filepath is not None:
                modifier.tlk_filepath = self._resolve_source_file_path(
                    tlk_filepath,
                    "TLK source file",
                )

        tlk_patches.append(patches_tlk)

        female_dialog_filename = "dialogf.tlk"
        female_dialog_file = self._resolve_relative_file_within(
            self.game_path,
            female_dialog_filename,
            "female dialog TLK",
        )
        if not female_dialog_file.safe_isfile():
            female_dialog_file = None

        if female_dialog_file is not None:
            female_tlk_patches: ModificationsTLK = deepcopy(patches_tlk)
            female_tlk_patches.saveas = female_dialog_filename
            female_tlk_patches.store_memory = False

            female_source_folder = self._resolve_relative_folder_within(
                self.patch_data_path,
                female_tlk_patches.sourcefolder,
                "female TLK source folder",
            )
            female_source_file = self._resolve_relative_file_within(
                female_source_folder,
                female_tlk_patches.sourcefile_f,
                "female TLK source file",
            )
            if not female_source_file.safe_isfile():
                female_source_file = None
            if female_source_file is not None:
                female_tlk_patches.sourcefile = female_tlk_patches.sourcefile_f
                for modifier in female_tlk_patches.modifiers:
                    if isinstance(modifier, MergeTLK):
                        modifier.tlk_filepath = female_source_file
            else:
                female_tlk_patches.modifiers = [
                    modifier
                    for modifier in female_tlk_patches.modifiers
                    if not isinstance(modifier, MergeTLK)
                ]

            if female_tlk_patches.modifiers:
                tlk_patches.append(female_tlk_patches)

        return tlk_patches
