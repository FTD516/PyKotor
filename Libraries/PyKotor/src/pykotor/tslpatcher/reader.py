from __future__ import annotations

import base64

from configparser import ConfigParser, ParsingError
from typing import TYPE_CHECKING

from pykotor.common.geometry import Vector3, Vector4
from pykotor.common.language import LocalizedString
from pykotor.common.misc import CaseInsensitiveDict, ResRef
from pykotor.common.stream import BinaryReader
from pykotor.resource.formats.gff import GFFFieldType, GFFList, GFFStruct
from pykotor.resource.formats.ssf import SSFSound
from pykotor.tools.encoding import decode_bytes_with_fallbacks
from pykotor.tools.path import CaseAwarePath
from pykotor.tslpatcher.config import LogLevel
from pykotor.tslpatcher.logger import PatchLogger
from pykotor.tslpatcher.memory import NoTokenUsage, TokenUsage2DA, TokenUsageTLK
from pykotor.tslpatcher.mods.gff import (
    AddFieldGFF,
    AddStructToListGFF,
    FieldValue2DAMemory,
    FieldValueConstant,
    FieldValueListIndex,
    FieldValueRaw,
    FieldValueTLKMemory,
    LocalizedStringDelta,
    Memory2DAModifierGFF,
    ModificationsGFF,
    ModifyFieldGFF,
)
from pykotor.tslpatcher.mods.install import InstallFile
from pykotor.tslpatcher.mods.ncs import ModificationsNCS
from pykotor.tslpatcher.mods.nss import ModificationsNSS
from pykotor.tslpatcher.mods.ssf import ModificationsSSF, ModifySSF
from pykotor.tslpatcher.mods.tlk import MergeTLK, ModificationsTLK, ModifyTLK
from pykotor.tslpatcher.mods.twoda import (
    AddColumn2DA,
    AddRow2DA,
    ChangeRow2DA,
    CopyRow2DA,
    Modifications2DA,
    RowValue2DAMemory,
    RowValueConstant,
    RowValueHigh,
    RowValueRowCell,
    RowValueRowIndex,
    RowValueRowLabel,
    RowValueTLKMemory,
    Target,
    TargetType,
)
from pykotor.tslpatcher.namespaces import PatcherNamespace
from utility.misc import is_float, is_int
from utility.system.path import Path, PurePath, PureWindowsPath

if TYPE_CHECKING:
    import os

    from typing_extensions import Literal

    from pykotor.tslpatcher.config import PatcherConfig
    from pykotor.tslpatcher.memory import TokenUsage
    from pykotor.tslpatcher.mods.gff import (
        FieldValue,
        ModifyGFF,
    )
    from pykotor.tslpatcher.mods.twoda import (
        Modify2DA,
        RowValue,
    )

SECTION_NOT_FOUND_ERROR = "The [{}] section was not found in the ini"
REFERENCES_TRACEBACK_MSG = ", referenced by '{}={}' in [{}]"


class NamespaceReader:
    """Responsible for reading and loading namespaces from the namespaces.ini file."""

    def __init__(self, ini: ConfigParser):
        self.ini: ConfigParser = ini
        self.namespaces: list[PatcherNamespace] = []

    @classmethod
    def from_filepath(cls, path: os.PathLike | str) -> list[PatcherNamespace]:
        ini = ConfigParser(
            delimiters=("="),
            allow_no_value=True,
            strict=False,
            interpolation=None,
        )
        # use case insensitive keys
        ini.optionxform = lambda optionstr: optionstr.lower()  # type: ignore[method-assign]

        ini.read_string(decode_bytes_with_fallbacks(BinaryReader.load_file(path)))
        return NamespaceReader(ini).load()

    def load(self) -> list[PatcherNamespace]:  # Case-insensitive access to section
        namespaces_section_name: str | None = next((section for section in self.ini.sections() if section.lower() == "namespaces"), None)
        if namespaces_section_name is None:
            raise KeyError(SECTION_NOT_FOUND_ERROR.format("Namespaces"))

        namespace_ids: CaseInsensitiveDict[str] = CaseInsensitiveDict(self.ini[namespaces_section_name])
        namespaces: list[PatcherNamespace] = []

        for key, namespace_id in namespace_ids.items():
            # Case-insensitive access to namespace_id
            namespace_section_key: str | None = next((section for section in self.ini.sections() if section.lower() == namespace_id.lower()), None)
            if namespace_section_key is None:
                msg = f"The '[{namespace_id}]' section was not found in the 'namespaces.ini' file, referenced by '{key}={namespace_id}' in [{namespaces_section_name}]."
                raise KeyError(msg)

            this_namespace_section = CaseInsensitiveDict(self.ini[namespace_section_key])

            # required
            ini_filename: str = this_namespace_section["IniName"]
            info_filename: str = this_namespace_section["InfoName"]
            namespace = PatcherNamespace(ini_filename, info_filename)

            # optional
            namespace.data_folderpath = PurePath(this_namespace_section.get("DataPath", ""))
            namespace.name = this_namespace_section.get("Name", "").strip()
            namespace.description = this_namespace_section.get("Description", "")

            namespace.namespace_id = namespace_section_key
            namespaces.append(namespace)

        return namespaces


class ConfigReader:
    def __init__(
        self,
        ini: ConfigParser,
        mod_path: os.PathLike | str,
        logger: PatchLogger | None = None,
        tslpatchdata_path: os.PathLike | str | None = None,
    ):
        self.previously_parsed_sections = set()
        self.ini: ConfigParser = ini
        self.mod_path = CaseAwarePath.get_case_sensitive_path(mod_path)
        self.tslpatchdata_path = (
            None
            if tslpatchdata_path is None
            else CaseAwarePath.get_case_sensitive_path(tslpatchdata_path)
        )  # path to the tslpatchdata, optional but we'll use it here for the nwnnsscomp.exe if it exists.
        self.patch_data_path = self.tslpatchdata_path or self.mod_path
        self.config: PatcherConfig
        self.log: PatchLogger = logger or PatchLogger()

    @classmethod
    def from_filepath(
        cls,
        file_path: os.PathLike | str,
        logger: PatchLogger | None = None,
        **kwargs,
    ):
        """Load PatcherConfig from an INI file path.

        Args:
        ----
            file_path: The path to the INI file.
            logger: Optional logger instance.

        Returns:
        -------
            A PatcherConfig instance loaded from the file.

        Processing Logic:
        ----------------
            - Resolve the file path and load its contents
            - Parse the INI text into a ConfigParser
            - Initialize a PatcherConfig instance
            - Populate its config attribute from the ConfigParser
            - Return the initialized instance
        """
        from pykotor.tslpatcher.config import PatcherConfig  # noqa: PLC0415 Prevent circular imports

        resolved_file_path: Path = Path.pathify(file_path).resolve()

        ini = ConfigParser(
            delimiters=("="),
            allow_no_value=True,
            strict=False,
            interpolation=None,
        )

        # Use case-sensitive keys
        ini.optionxform = lambda optionstr: optionstr if optionstr.strip() else optionstr.strip()  # type: ignore[method-assign]
        try:
            ini.read_string(decode_bytes_with_fallbacks(BinaryReader.load_file(resolved_file_path)))
        except ParsingError as e:
            e.source = str(resolved_file_path)
            raise e  # noqa: TRY201  # don't `raise from e` here!

        instance = cls(ini, resolved_file_path.parent, logger)
        instance.config = PatcherConfig()
        return instance

    def load(self, config: PatcherConfig) -> PatcherConfig:
        self.config = config
        self.previously_parsed_sections = set()

        self.load_settings()
        self.load_tlk_list()
        self.load_install_list()
        self.load_2da_list()
        self.load_gff_list()
        self.load_hack_list()
        self.load_compile_list()
        self.load_ssf_list()
        self.log.add_note("The ConfigReader finished loading the INI")
        all_sections_set = set(self.ini.sections())
        orphaned_sections: set[str] = all_sections_set - self.previously_parsed_sections
        if len(orphaned_sections):
            orphaned_sections_str: str = "\n".join(orphaned_sections)
            self.log.add_note(f"There are some orphaned ini sections found in the changes:\n{orphaned_sections_str}")

        return self.config

    def get_section_name(self, section_name: str) -> str | None:
        """Resolves the case-insensitive section name string if found and returns the case-sensitive correct section name."""
        s: str | None = next(
            (section for section in self.ini.sections() if section.lower() == section_name.lower()),
            None,
        )
        if s is not None:
            self.previously_parsed_sections.add(s)
        return s

    def load_settings(self):
        """Loads [Settings] from ini configuration into memory."""
        settings_section: str | None = self.get_section_name("settings")
        if settings_section is None:
            self.log.add_warning("[Settings] section missing from ini.")
            return

        self.log.add_note("Loading [Settings] section from ini...")
        settings_ini = CaseInsensitiveDict(self.ini[settings_section])

        self.config.window_title = settings_ini.get("WindowCaption", "")
        self.config.confirm_message = settings_ini.get("ConfirmMessage", "")
        required_files: dict[str, tuple[str, ...]] = {}
        required_messages: dict[str, str] = {}
        required_order: list[str] = []
        for key, value in settings_ini.items():
            lower_key = key.lower()
            if lower_key.startswith("requiredmsg"):
                suffix = key[len("requiredmsg"):]
                if suffix and (not suffix.isascii() or not suffix.isdigit()):
                    raise ValueError(f"Key '{key}' improperly defined in settings ini. Expected (Required) or (RequiredMsg)")
                required_messages[suffix] = (value or "").strip()

            elif lower_key.startswith("required"):
                suffix = key[len("required"):]
                if suffix and (not suffix.isascii() or not suffix.isdigit()):
                    raise ValueError(f"Key '{key}' improperly defined in settings ini. Expected (Required) or (RequiredMsg)")
                filenames = tuple(filename.strip() for filename in (value or "").split(",") if filename.strip())
                if not filenames:
                    continue
                required_files[suffix] = filenames
                required_order.append(suffix)

        for suffix in required_order:
            self.config.required_files.append(required_files[suffix])
            self.config.required_messages.append(required_messages.get(suffix, ""))
        self.config.save_processed_scripts = int(settings_ini.get("SaveProcessedScripts", 0))
        self.config.script_compiler_flags = settings_ini.get("ScriptCompilerFlags", "") or ""
        self.config.log_level = LogLevel(int(settings_ini.get("LogLevel", LogLevel.WARNINGS.value)))

        # HoloPatcher optional
        self.config.ignore_file_extensions = bool(settings_ini.get("IgnoreExtensions")) or False

        lookup_game_number: str | None = settings_ini.get("LookupGameNumber")
        if lookup_game_number:
            lookup_game_number = lookup_game_number.strip()
            if lookup_game_number not in {"1", "2"}:
                msg = f"Invalid: 'LookupGameNumber={lookup_game_number}' in [Settings], must be 1 or 2 representing the KOTOR game."
                raise ValueError(msg)
            self.config.game_number = int(lookup_game_number)
        else:
            self.config.game_number = None

    def load_install_list(self):
        """Loads [InstallList] from ini configuration into memory.

        Processing Logic:
        ----------------
            - Gets [InstallList] section from ini
            - Loops through section items getting foldername and filenames
            - Gets section for each filename
            - Creates InstallFile object for each filename
            - Adds InstallFile to config install list
            - Optionally loads additional vars from filename section
        """
        install_list_section: str | None = self.get_section_name("InstallList")
        if install_list_section is None:
            self.log.add_note("[InstallList] section missing from ini.")
            return

        self.log.add_note("Loading [InstallList] patches from ini...")
        for folder_key, foldername in self.ini[install_list_section].items():
            foldername_section: str | None = self.get_section_name(folder_key)
            if foldername_section is None:
                raise KeyError(SECTION_NOT_FOUND_ERROR.format(foldername) + REFERENCES_TRACEBACK_MSG.format(folder_key, foldername, install_list_section))

            folder_section_dict = CaseInsensitiveDict(self.ini[foldername_section])
            sourcefolder: str = folder_section_dict.pop("!SourceFolder", ".")
            folder_section_dict.pop("!OverrideType", None)
            for file_key, filename in folder_section_dict.items():
                file_install = InstallFile(
                    filename,
                    replace_existing=file_key.lower().startswith("replace"),
                )
                file_install.destination = foldername
                file_install.sourcefolder = sourcefolder
                self.config.install_list.append(file_install)

                # Optional according to TSLPatcher readme
                file_section_name: str | None = self.get_section_name(filename)
                if file_section_name is not None:
                    file_section_dict = CaseInsensitiveDict(self.ini[file_section_name])
                    file_install.pop_tslpatcher_vars(file_section_dict, foldername, sourcefolder)

    def load_tlk_list(self):
        """Loads TLK patches from the ini file into memory.

        Processing Logic:
        ----------------
            - Parses the [TLKList] section to get TLK patch entries
            - Handles different patch syntaxes like file replacements, string references etc
            - Builds TLK merge and modification operations and adds them to the patch list
            - Raises errors for invalid syntax or missing files
        """
        tlk_list_section: str | None = self.get_section_name("tlklist")
        if tlk_list_section is None:
            self.log.add_note("[TLKList] section missing from ini.")
            return

        self.log.add_note("Loading [TLKList] patches from ini...")
        tlk_list_edits = CaseInsensitiveDict(self.ini[tlk_list_section])

        default_destination = tlk_list_edits.pop("!DefaultDestination", ModificationsTLK.DEFAULT_DESTINATION)
        default_sourcefolder = tlk_list_edits.pop("!DefaultSourceFolder", ".")
        self.config.patches_tlk.pop_tslpatcher_vars(tlk_list_edits, default_destination, default_sourcefolder)

        modifier_dict: dict[int, dict[str, str]] = {}
        source_tlk_merge: MergeTLK | None = None
        syntax_error_caught = False

        def parse_tlk_index(raw_value: str | None, description: str) -> int:
            if (
                not isinstance(raw_value, str)
                or not raw_value
                or not raw_value.isascii()
                or not raw_value.isdigit()
            ):
                msg = f"Invalid {description} '{raw_value}': expected non-negative ASCII decimal digits"
                raise ValueError(msg)
            return int(raw_value)

        def process_tlk_entries(
            tlk_filename: str,
            dialog_tlk_index: int,
            mod_tlk_index: int,
            *,
            is_replacement: bool,
        ):
            modifier = ModifyTLK(dialog_tlk_index, is_replacement)
            modifier.mod_index = mod_tlk_index
            modifier.tlk_filepath = CaseAwarePath.get_case_sensitive_path(
                self.patch_data_path / self.config.patches_tlk.sourcefolder / tlk_filename,
            )
            self.config.patches_tlk.modifiers.append(modifier)

        for key, value in tlk_list_edits.items():
            lower_key: str = key.lower()
            replace_file: bool = lower_key.startswith("replace")
            append_file: bool = lower_key.startswith("append")
            try:
                if lower_key.startswith("strref"):
                    token_id = parse_tlk_index(key[6:], "StrRef token")
                    source_stringref = parse_tlk_index(value, "source TLK index")
                    if source_tlk_merge is None:
                        source_tlk_merge = MergeTLK(
                            CaseAwarePath.get_case_sensitive_path(
                                self.patch_data_path
                                / self.config.patches_tlk.sourcefolder
                                / self.config.patches_tlk.sourcefile,
                            ),
                        )
                        self.config.patches_tlk.modifiers.append(source_tlk_merge)
                    source_tlk_merge.add_mapping(token_id, source_stringref)
                elif replace_file or append_file:
                    next_section_name = self.get_section_name(value)
                    if next_section_name is None:
                        syntax_error_caught = True
                        raise ValueError(SECTION_NOT_FOUND_ERROR.format(value) + REFERENCES_TRACEBACK_MSG.format(key, value, tlk_list_section))  # noqa: TRY301

                    next_section_dict = CaseInsensitiveDict(self.ini[next_section_name])
                    self.config.patches_tlk.pop_tslpatcher_vars(next_section_dict, default_destination, default_sourcefolder)

                    for raw_dialog_tlk_index, raw_mod_tlk_index in zip(
                        self.ini[next_section_name].keys(),
                        self.ini[next_section_name].values(),
                    ):
                        dialog_index_text = (
                            raw_dialog_tlk_index[6:]
                            if raw_dialog_tlk_index.lower().startswith("strref")
                            else raw_dialog_tlk_index
                        )
                        mod_index_text = (
                            raw_mod_tlk_index[6:]
                            if isinstance(raw_mod_tlk_index, str) and raw_mod_tlk_index.lower().startswith("strref")
                            else raw_mod_tlk_index
                        )
                        dialog_tlk_index = parse_tlk_index(dialog_index_text, "destination TLK index")
                        mod_tlk_index = parse_tlk_index(mod_index_text, "source TLK index")
                        process_tlk_entries(
                            tlk_filename=next_section_name,
                            dialog_tlk_index=dialog_tlk_index,
                            mod_tlk_index=mod_tlk_index,
                            is_replacement=replace_file,
                        )
                elif "\\" in lower_key or "/" in lower_key:
                    delimiter: Literal["\\", "/"] = "\\" if "\\" in lower_key else "/"
                    token_id_str, property_name = lower_key.split(delimiter)
                    token_id = parse_tlk_index(token_id_str, "destination TLK index")

                    if token_id not in modifier_dict:
                        modifier_dict[token_id] = {"text": "", "voiceover": ""}

                    if property_name == "text":
                        modifier = ModifyTLK(token_id, is_replacement=True)
                        modifier.text = value
                        self.config.patches_tlk.modifiers.append(modifier)
                    elif property_name == "sound":
                        modifier = ModifyTLK(token_id, is_replacement=True)
                        modifier.sound = ResRef(value)
                        self.config.patches_tlk.modifiers.append(modifier)
                    else:
                        syntax_error_caught = True
                        msg = f"Invalid [TLKList] syntax: '{key}={value}'! Expected '{key}' to be one of ['Sound', 'Text']"
                        raise ValueError(msg)  # noqa: TRY301
                else:
                    syntax_error_caught = True
                    msg = f"Invalid syntax found in [TLKList] '{key}={value}'! Expected '{key}' to be one of ['AppendFile', 'ReplaceFile', '!SourceFile', 'StrRef', 'Text', 'Sound']"
                    raise ValueError(msg)  # noqa: TRY301

            except ValueError as e:
                if syntax_error_caught:
                    raise
                msg = f"Could not parse '{key}={value}' in [TLKList]"
                raise ValueError(msg) from e

    def load_2da_list(self):
        """Load 2D array patches from ini file into memory.

        Processing Logic:
        ----------------
            - Get the section name for the [2DAList] section
            - Load the section into a dictionary
            - Pop the default destination key
            - Iterate through each identifier and file
                - Get the section for the file
                - Create a Modifications2DA object for the file
                - Load the section into a dictionary and populate the object
                - Append the object to the config patches list
                - Iterate through each key and modification ID
                    - Get the section for the ID
                    - Load the section into a dictionary
                    - Discern and add the modifier to the file object.
        """
        twoda_section_name: str | None = self.get_section_name("2DAList")
        if twoda_section_name is None:
            self.log.add_note("[2DAList] section missing from ini.")
            return

        self.log.add_note("Loading [2DAList] patches from ini...")

        twoda_section_dict = CaseInsensitiveDict(self.ini[twoda_section_name])
        default_destination: str = twoda_section_dict.pop("!DefaultDestination", Modifications2DA.DEFAULT_DESTINATION)
        default_source_folder = twoda_section_dict.pop("!DefaultSourceFolder", ".")
        for identifier, file in twoda_section_dict.items():
            file_section: str | None = self.get_section_name(file)
            if file_section is None:
                raise KeyError(SECTION_NOT_FOUND_ERROR.format(file) + REFERENCES_TRACEBACK_MSG.format(identifier, file, twoda_section_name))

            modifications = Modifications2DA(file)
            file_section_dict = CaseInsensitiveDict(self.ini[file_section])
            modifications.pop_tslpatcher_vars(file_section_dict, default_destination, default_source_folder)

            self.config.patches_2da.append(modifications)

            for key, modification_id in file_section_dict.items():
                if not key or not modification_id:
                    continue

                next_section_name: str | None = self.get_section_name(modification_id)
                modification_ids_dict = (
                    CaseInsensitiveDict()
                    if next_section_name is None
                    else CaseInsensitiveDict(self.ini[next_section_name])
                )
                manipulation: Modify2DA | None = self.discern_2da(key, modification_id, modification_ids_dict)
                if not manipulation:
                    continue
                modifications.modifiers.append(manipulation)

    def load_ssf_list(self):
        """Load SSF patches without allowing one invalid entry to abort the file."""
        ssf_list_section = self.get_section_name("SSFList")
        if ssf_list_section is None:
            self.log.add_note("[SSFList] section missing from ini.")
            return

        self.log.add_note("Loading [SSFList] patches from ini...")
        ssf_section_dict = CaseInsensitiveDict(self.ini[ssf_list_section])
        default_destination = ssf_section_dict.pop("!DefaultDestination", ModificationsSSF.DEFAULT_DESTINATION)
        default_source_folder = ssf_section_dict.pop("!DefaultSourceFolder", ".")

        for identifier, file in ssf_section_dict.items():
            ssf_file_section = self.get_section_name(file)
            if ssf_file_section is None:
                raise KeyError(SECTION_NOT_FOUND_ERROR.format(file) + REFERENCES_TRACEBACK_MSG.format(identifier, file, ssf_list_section))

            modifications = ModificationsSSF(file, identifier.lower().startswith("replace"))
            self.config.patches_ssf.append(modifications)

            file_section_dict = CaseInsensitiveDict(self.ini[ssf_file_section])
            modifications.pop_tslpatcher_vars(file_section_dict, default_destination, default_source_folder)

            for name, value in file_section_dict.items():
                try:
                    sound = self.resolve_tslpatcher_ssf_sound(name)
                except KeyError:
                    self.log.add_error(f"Unknown SSF sound label '{name}' in [{ssf_file_section}]; skipping entry.")
                    continue
                raw_value = value or ""
                suffix = raw_value[6:]
                if raw_value[:6].lower() == "strref" and suffix.isascii() and suffix.isdigit():
                    stringref = TokenUsageTLK(int(suffix))
                else:
                    suffix = raw_value[9:]
                    if raw_value.startswith("2DAMEMORY") and suffix.isascii() and suffix.isdigit():
                        stringref = TokenUsage2DA(int(suffix))
                    else:
                        stringref = NoTokenUsage(raw_value)
                modifications.modifiers.append(ModifySSF(sound, stringref))

    def load_gff_list(self):
        """Load GFF patches while keeping each field operation independent."""
        gff_list_section = self.get_section_name("GFFList")
        if gff_list_section is None:
            self.log.add_note("[GFFList] section missing from ini.")
            return

        self.log.add_note("Loading [GFFList] patches from ini...")
        gff_section_dict = CaseInsensitiveDict(self.ini[gff_list_section])
        default_destination = gff_section_dict.pop("!DefaultDestination", ModificationsGFF.DEFAULT_DESTINATION)
        default_source_folder = gff_section_dict.pop("!DefaultSourceFolder", ".")

        for identifier, file in gff_section_dict.items():
            file_section_name = self.get_section_name(file)
            if file_section_name is None:
                raise KeyError(SECTION_NOT_FOUND_ERROR.format(file) + REFERENCES_TRACEBACK_MSG.format(identifier, file, gff_list_section))

            modifications = ModificationsGFF(file, identifier.lower().startswith("replace"))
            self.config.patches_gff.append(modifications)

            file_section_dict = CaseInsensitiveDict(self.ini[file_section_name])
            modifications.pop_tslpatcher_vars(file_section_dict, default_destination, default_source_folder)

            for key, value in file_section_dict.items():
                try:
                    modifier: ModifyGFF | None = None
                    raw_value = value or ""
                    if key.startswith("AddField") and len(key) > len("AddField"):
                        next_gff_section = self.get_section_name(raw_value)
                        if next_gff_section is None:
                            self.log.add_error(SECTION_NOT_FOUND_ERROR.format(raw_value) + REFERENCES_TRACEBACK_MSG.format(key, raw_value, file_section_name))
                            continue
                        modifier = self.add_field_gff(next_gff_section, CaseInsensitiveDict(self.ini[next_gff_section]))

                    elif key.startswith("2DAMEMORY") and key[9:].isascii() and key[9:].isdigit():
                        token_id = int(key[9:])
                        if raw_value == "!FieldPath":
                            modifier = Memory2DAModifierGFF(file_section_name, PureWindowsPath(""), token_id)
                        elif raw_value.startswith("2DAMEMORY") and raw_value[9:].isascii() and raw_value[9:].isdigit():
                            modifier = Memory2DAModifierGFF(file_section_name, PureWindowsPath(""), token_id, int(raw_value[9:]))
                        else:
                            modifier = self.modify_field_gff(file_section_name, key, raw_value)
                    else:
                        modifier = self.modify_field_gff(file_section_name, key, raw_value)

                    if modifier is not None:
                        modifications.modifiers.append(modifier)
                except (KeyError, TypeError, ValueError) as exc:
                    self.log.add_error(f"Unable to parse GFF operation '{key}={value}' in [{file_section_name}]: {exc}")

    def load_compile_list(self):
        """Loads patches from the [CompileList] section of the ini file.

        Processing Logic:
        ----------------
            - Parses the [CompileList] section of the ini file into a dictionary
            - Sets a default destination from an optional key
            - Loops through each identifier/file pair
                - Creates a ModificationsNSS object
                - Looks for an optional section for the file
                - Passes any values to populate the patch
            - Adds each patch to the config patches list
        """
        compilelist_section: str | None = self.get_section_name("CompileList")
        if compilelist_section is None:
            self.log.add_note("[CompileList] section missing from ini.")
            return

        self.log.add_note("Loading [CompileList] patches from ini...")
        compilelist_section_dict = CaseInsensitiveDict(self.ini[compilelist_section])
        default_destination: str = compilelist_section_dict.pop("!DefaultDestination", ModificationsNSS.DEFAULT_DESTINATION)
        default_source_folder = compilelist_section_dict.pop("!DefaultSourceFolder", ".")

        nwnnsscomp_exepath = CaseAwarePath.get_case_sensitive_path(
            self.patch_data_path / default_source_folder / "nwnnsscomp.exe",
        )
        if not nwnnsscomp_exepath.safe_isfile():
            nwnnsscomp_exepath = (
                None
                if self.tslpatchdata_path is None
                else CaseAwarePath.get_case_sensitive_path(self.tslpatchdata_path / "nwnnsscomp.exe")
            )  # TSLPatcher default

        for identifier, file in compilelist_section_dict.items():
            replace: bool = identifier.lower().startswith("replace")
            modifications = ModificationsNSS(file, replace)
            modifications.destination = default_destination
            modifications.sourcefolder = default_source_folder

            optional_file_section_name: str | None = self.get_section_name(file)
            if optional_file_section_name is not None:
                file_section_dict = CaseInsensitiveDict(self.ini[optional_file_section_name])
                modifications.pop_tslpatcher_vars(file_section_dict, default_destination, default_source_folder)

            modifications.nwnnsscomp_path = nwnnsscomp_exepath
            modifications.compiler_flags = self.config.script_compiler_flags
            self.config.patches_nss.append(modifications)

    def load_hack_list(self):
        """Loads [HACKList] patches from ini file into memory.

        Processing Logic:
        ----------------
            1. Gets the "[HACKList]" section name from the ini file
            2. Loops through each identifier and file in the section
            3. Creates a ModificationsNCS object for each file
            4. Populates the object with offset/value pairs from the file's section
            5. Adds the populated object to the config patches list
        """
        hacklist_section: str | None = self.get_section_name("HACKList")
        if hacklist_section is None:
            self.log.add_note("[HACKList] section missing from ini.")
            return

        self.log.add_note("Loading [HACKList] patches from ini...")
        hacklist_section_dict = CaseInsensitiveDict(self.ini[hacklist_section])
        default_destination: str = hacklist_section_dict.pop("!DefaultDestination", ModificationsNCS.DEFAULT_DESTINATION)
        default_source_folder = hacklist_section_dict.pop("!DefaultSourceFolder", ".")

        for identifier, file in hacklist_section_dict.items():
            replace: bool = identifier.lower().startswith("replace")
            modifications = ModificationsNCS(file, replace)

            file_section_name: str | None = self.get_section_name(file)
            if file_section_name is None:
                self.log.add_error(
                    SECTION_NOT_FOUND_ERROR.format(file)
                    + REFERENCES_TRACEBACK_MSG.format(identifier, file, hacklist_section),
                )
                continue

            file_section_dict = CaseInsensitiveDict(self.ini[file_section_name])
            modifications.pop_tslpatcher_vars(file_section_dict, default_destination, default_source_folder)

            for offset_str, value_str in file_section_dict.items():
                raw_offset = offset_str.strip()
                raw_value = (value_str or "").strip()
                if not raw_offset.isascii() or not raw_offset.isdigit():
                    self.log.add_warning(
                        f"Ignoring invalid HACKList offset '{offset_str}' in [{file_section_name}].",
                    )
                    continue

                offset = int(raw_offset)
                if offset > 0x7FFFFFFF:
                    self.log.add_warning(
                        f"Ignoring out-of-range HACKList offset '{offset_str}' in [{file_section_name}].",
                    )
                    continue

                lower_value = raw_value.lower()
                if lower_value.startswith("strref"):
                    token_id = raw_value[6:]
                    if not token_id.isascii() or not token_id.isdigit() or int(token_id) > 0x7FFFFFFF:
                        self.log.add_warning(
                            f"Ignoring invalid HACKList value '{value_str}' at offset {offset_str} in [{file_section_name}].",
                        )
                        continue
                    modifications.hackdata.append(("StrRef", offset, int(token_id)))
                elif raw_value.startswith("2DAMEMORY"):
                    token_id = raw_value[9:]
                    if token_id.isascii() and token_id.isdigit():
                        if int(token_id) > 0x7FFFFFFF:
                            self.log.add_warning(
                                f"Ignoring invalid HACKList value '{value_str}' at offset {offset_str} in [{file_section_name}].",
                            )
                            continue
                        modifications.hackdata.append(("2DAMEMORY", offset, int(token_id)))
                    else:
                        modifications.hackdata.append(("2DAMEMORY", offset, 0))
                else:
                    if not raw_value.isascii() or not raw_value.isdigit() or int(raw_value) > 0x7FFFFFFF:
                        self.log.add_warning(
                            f"Ignoring invalid HACKList value '{value_str}' at offset {offset_str} in [{file_section_name}].",
                        )
                        continue
                    modifications.hackdata.append(("VALUE", offset, int(raw_value)))

            self.config.patches_ncs.append(modifications)

    #################

    @classmethod
    def modify_field_gff(
        cls,
        identifier: str,
        key: str,
        str_value: str,
    ) -> ModifyFieldGFF:
        """Create a direct GFF field operation without committing to a destination type."""
        value: FieldValue = cls._deferred_gff_value(str_value)
        lower_key = key.lower()
        path_token_id: int | None = None

        if "(strref)" in lower_key:
            selector = lower_key.index("(strref)")
            value = FieldValueConstant(LocalizedStringDelta(cls._deferred_gff_value(str_value, GFFFieldType.Int32)))
            key = key[:selector]

        else:
            selector = key.find("(lang")
            if selector != -1 and key.endswith(")"):
                substring = key[selector + 5 : -1]
                if substring.isascii() and substring.isdigit():
                    substring_id = int(substring)
                    locstring = LocalizedStringDelta()
                    field_value = cls._deferred_gff_value(str_value, GFFFieldType.String)
                    if cls._requires_runtime_gff_resolution(str_value):
                        locstring.set_field_value(substring_id, field_value)
                    else:
                        language, gender = locstring.substring_pair(substring_id)
                        locstring.set_data(language, gender, cls.normalize_tslpatcher_crlf(str_value))
                    value = FieldValueConstant(locstring)
                    key = key[:selector]

        if key.startswith("2DAMEMORY") and key[9:].isascii() and key[9:].isdigit():
            path_token_id = int(key[9:])
            key = ""

        return ModifyFieldGFF(PureWindowsPath(key), value, identifier, path_token_id=path_token_id)

    @classmethod
    def _deferred_gff_value(
        cls,
        raw_value: str,
        field_type: GFFFieldType | None = None,
    ) -> FieldValue:
        memory_value = cls.field_value_from_memory(raw_value)
        if memory_value is not None:
            return memory_value

        parsed_value: FieldValue | None = None
        try:
            parsed_value = cls.field_value_from_type(raw_value, field_type) if field_type is not None else cls.field_value_from_unknown(raw_value)
        except (TypeError, ValueError):
            pass

        preview = parsed_value.stored if isinstance(parsed_value, FieldValueConstant) else raw_value
        return FieldValueRaw(raw_value, preview)

    @staticmethod
    def _requires_runtime_gff_resolution(raw_value: str) -> bool:
        return raw_value.startswith("2DAMEMORY") or (
            raw_value[:6].lower() == "strref"
            and raw_value[6:].isascii()
            and raw_value[6:].isdigit()
        )

    def add_field_gff(
        self,
        identifier: str,
        ini_data: CaseInsensitiveDict[str],
        current_path: PureWindowsPath | None = None,
    ) -> ModifyGFF:
        """Parse one AddField section while preserving nested operation order."""
        raw_field_type = ini_data.get("FieldType")
        if raw_field_type is None:
            raise ValueError(f"FieldType must be set in [{identifier}].")
        field_type = self.resolve_tslpatcher_gff_field_type(raw_field_type)

        label_value = ini_data.get("Label", "")
        label = "" if label_value is None else label_value.strip()
        has_explicit_path = "Path" in ini_data
        raw_path = ini_data.get("Path", "") or ""
        path = PureWindowsPath(raw_path.strip())
        relative_path = current_path is not None and not has_explicit_path

        value = self._get_addfield_value(ini_data, field_type, identifier)
        modifiers: list[ModifyGFF] = []

        for key, iterated_value in ini_data.items():
            raw_iterated_value = iterated_value or ""
            if key.startswith("2DAMEMORY") and key[9:].isascii() and key[9:].isdigit():
                token_id = int(key[9:])
                if raw_iterated_value == "ListIndex":
                    modifiers.append(
                        Memory2DAModifierGFF(
                            identifier,
                            PureWindowsPath(""),
                            token_id,
                            relative_path=True,
                            store_list_index=True,
                        )
                    )
                elif raw_iterated_value == "!FieldPath":
                    modifiers.append(
                        Memory2DAModifierGFF(
                            identifier,
                            PureWindowsPath(""),
                            token_id,
                            relative_path=True,
                        )
                    )
                elif raw_iterated_value.startswith("2DAMEMORY") and raw_iterated_value[9:].isascii() and raw_iterated_value[9:].isdigit():
                    modifiers.append(
                        Memory2DAModifierGFF(
                            identifier,
                            PureWindowsPath(""),
                            token_id,
                            int(raw_iterated_value[9:]),
                            relative_path=True,
                        )
                    )
                continue

            if key.startswith("AddField") and len(key) > len("AddField"):
                next_section_name = self.get_section_name(raw_iterated_value)
                if next_section_name is None:
                    self.log.add_error(SECTION_NOT_FOUND_ERROR.format(raw_iterated_value) + REFERENCES_TRACEBACK_MSG.format(key, raw_iterated_value, identifier))
                    continue
                try:
                    nested_modifier = self.add_field_gff(
                        next_section_name,
                        CaseInsensitiveDict(self.ini[next_section_name]),
                        current_path=PureWindowsPath(""),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    self.log.add_error(f"Unable to parse nested AddField section [{next_section_name}]: {exc}")
                    continue
                modifiers.append(nested_modifier)

        if field_type is GFFFieldType.Struct and not label:
            return AddStructToListGFF(
                identifier,
                value,
                path,
                modifiers=modifiers,
                relative_path=relative_path,
            )
        if not label:
            raise ValueError(f"Label must be set for FieldType={field_type.name} in [{identifier}].")
        return AddFieldGFF(
            identifier,
            label,
            field_type,
            value,
            path,
            modifiers,
            relative_path=relative_path,
        )

    @classmethod
    def _get_addfield_value(
        cls,
        ini_section_dict: CaseInsensitiveDict[str],
        field_type: GFFFieldType,
        identifier: str,
    ) -> FieldValue:
        """Return an AddField value that retains the original INI text for apply-time validation."""
        if field_type is GFFFieldType.LocalizedString:
            return cls.field_value_from_localized_string(ini_section_dict)
        if field_type is GFFFieldType.List:
            return FieldValueConstant(GFFList())
        if field_type is GFFFieldType.Struct:
            raw_type_id = ini_section_dict.get("TypeId", "0") or "0"
            if raw_type_id.lower() == "listindex":
                return FieldValueListIndex("listindex")
            type_id = int(raw_type_id) if raw_type_id.isascii() and raw_type_id.isdigit() else 0
            return FieldValueRaw(raw_type_id, GFFStruct(type_id))
        if "Value" in ini_section_dict:
            return cls._deferred_gff_value(ini_section_dict.get("Value", "") or "", field_type)
        if field_type is GFFFieldType.ResRef:
            return FieldValueRaw("", ResRef.from_blank())
        if field_type is GFFFieldType.String:
            return FieldValueRaw("", "")
        if field_type is GFFFieldType.Binary:
            return FieldValueRaw("", b"")
        return FieldValueRaw("")

    @classmethod
    def field_value_from_localized_string(cls, ini_section_dict: CaseInsensitiveDict) -> FieldValueConstant:
        """Parse a localized-string delta while deferring token substitution until apply time."""
        raw_stringref = ini_section_dict.get("StrRef", "-1") or "-1"
        delta = LocalizedStringDelta(cls._deferred_gff_value(raw_stringref, GFFFieldType.Int32))

        for substring, text in ini_section_dict.items():
            if not substring.startswith("lang"):
                continue
            suffix = substring[4:]
            if not suffix.isascii() or not suffix.isdigit():
                continue

            raw_text = text or ""
            substring_id = int(suffix)
            if cls._requires_runtime_gff_resolution(raw_text):
                delta.set_field_value(substring_id, cls._deferred_gff_value(raw_text, GFFFieldType.String))
            else:
                language, gender = delta.substring_pair(substring_id)
                delta.set_data(language, gender, cls.normalize_tslpatcher_crlf(raw_text))

        return FieldValueConstant(delta)

    @staticmethod
    def field_value_from_memory(raw_value: str) -> FieldValue | None:
        """Return a GFF memory token only when the token grammar is valid."""
        suffix = raw_value[6:]
        if raw_value[:6].lower() == "strref" and suffix.isascii() and suffix.isdigit():
            return FieldValueTLKMemory(int(suffix))

        suffix = raw_value[9:]
        if raw_value.startswith("2DAMEMORY") and suffix.isascii() and suffix.isdigit():
            return FieldValue2DAMemory(int(suffix))

        return None

    @staticmethod
    def field_value_from_unknown(raw_value: str) -> FieldValue:
        """Extracts a field value from an unknown string representation.

            This section determines how to parse ini key/value pairs in gfflist such as:
                EntryList/0/RepliesList/0/TypeId=5

        Args:
        ----
            string_value: The string to parse.

        Returns:
        -------
            FieldValue: The parsed value represented as a FieldValue object.

        Processing Logic:
        ----------------
            - Checks if the value is already cached in memory
            - Tries to parse as int if possible
            - Otherwise parses as float by normalizing the string
            - Checks for Vector3 or Vector4 by counting pipe separators
            - Falls back to returning as string if no other type matches
            - Returns a FieldValueConstant wrapping the extracted value
        """
        # StrRef or 2DAMemory
        field_value_memory: FieldValue | None = ConfigReader.field_value_from_memory(raw_value)
        if field_value_memory is not None:
            return field_value_memory

        # Int
        if is_int(raw_value):
            return FieldValueConstant(int(raw_value))

        # Float
        parsed_float_str: str = ConfigReader.normalize_tslpatcher_float(raw_value)
        if is_float(parsed_float_str):  # float
            return FieldValueConstant(float(parsed_float_str))

        num_pipe_seps: int = raw_value.count("|")

        # String
        if not num_pipe_seps:
            return FieldValueConstant(ConfigReader.normalize_tslpatcher_crlf(raw_value))

        # Vector
        not_a_vector: bool = False
        components: list[float] = []
        for x in parsed_float_str.split("|"):
            if is_float(x):
                components.append(float(x))
                continue
            not_a_vector = True
            break

        # String containing the character '|'
        if not_a_vector:
            return FieldValueConstant(ConfigReader.normalize_tslpatcher_crlf(raw_value))

        # Three floats
        if num_pipe_seps == 2:
            return FieldValueConstant(Vector3(*components))

        # Four floats
        if num_pipe_seps == 3:
            return FieldValueConstant(Vector4(*components))

        msg = f"Cannot determine type/value from '{raw_value}'"
        raise ValueError(msg)

    @staticmethod
    def field_value_from_type(raw_value: str, field_type: GFFFieldType) -> FieldValue | None:
        # sourcery skip: assign-if-exp, reintroduce-else
        """Extracts field value from raw string based on field type.

        Args:
        ----
            raw_value (str): {Raw string value from file}
            field_type (GFFFieldType): {Field type enum}.

        Returns:
        -------
            FieldValue: {Field value object}

        Processing Logic:
        ----------------
            - Checks if value already exists in memory as a 2DAMEMORY or StrRef
            - Otherwise, converts raw_value to appropriate type based on field_type
            - Returns FieldValueConstant object wrapping extracted value.
        """
        field_value_memory: FieldValue | None = ConfigReader.field_value_from_memory(raw_value)
        if field_value_memory is not None:
            return field_value_memory

        components: list[float]
        value: ResRef | str | int | float | Vector3 | Vector4 | bytes | None = None

        if field_type.return_type() == ResRef:
            value = ResRef(raw_value)

        elif field_type.return_type() == str:
            value = ConfigReader.normalize_tslpatcher_crlf(raw_value)

        elif issubclass(field_type.return_type(), int):
            value = int(raw_value)

        elif issubclass(field_type.return_type(), float):
            value = float(ConfigReader.normalize_tslpatcher_float(raw_value))

        elif field_type.return_type() == Vector3:
            components = [float(ConfigReader.normalize_tslpatcher_float(axis)) for axis in raw_value.split("|")]
            value = Vector3(*components)

        elif field_type.return_type() == Vector4:
            components = [float(ConfigReader.normalize_tslpatcher_float(axis)) for axis in raw_value.split("|")]
            value = Vector4(*components)

        elif field_type.return_type() == bytes:
            if not raw_value.strip().replace("1", "").replace("0", ""):
                value = bytes(int(raw_value[i : i+8], 2) for i in range(0, len(raw_value), 8))
            elif raw_value.strip().lower().startswith("0x"):
                hex_string = raw_value[2:]
                if len(hex_string) % 2:
                    hex_string = f"0{hex_string}"
                value = bytes.fromhex(hex_string.strip())
            else:
                try:
                    value = base64.b64decode(raw_value)
                except Exception as e:  # noqa: BLE001
                    raise ValueError(f"The raw value for the binary field specified was invalid: '{raw_value}'") from e

        if value is None:
            return None

        return FieldValueConstant(value)
    #################

    def discern_2da(
        self,
        key: str,
        identifier: str,
        modifiers: CaseInsensitiveDict[str],
    ) -> Modify2DA | None:
        """Determines the type of 2DA modification based on the key.

        Args:
        ----
            key: str - The key identifying the type of modification
            identifier: str - The identifier of the 2DA (section name)
            modifiers: CaseInsensitiveDict[str] - Additional parameters for the modification

        Returns:
        -------
            Modify2DA | None - The 2DA modification object or None

        Processing Logic:
        ----------------
            - Preserves the modifier section's original entries
            - Parses the key to determine modification type
            - Constructs the modifier without eagerly normalizing its entries
            - Returns the modification object or None.
        """
        ordered_entries = [(modifier, "" if value is None else value) for modifier, value in modifiers.items()]

        if "AddRow" in key:
            return AddRow2DA(identifier, entries=ordered_entries)
        if "ChangeRow" in key:
            return ChangeRow2DA(identifier, entries=ordered_entries)
        if "AddColumn" in key:
            return AddColumn2DA(identifier, entries=ordered_entries)
        if "CopyRow" in key:
            return CopyRow2DA(identifier, entries=ordered_entries)

        self.log.add_warning(
            f"Could not parse key '{key}={identifier}', expecting one of ['AddRow', 'ChangeRow', 'AddColumn', 'CopyRow']",
        )
        return None

    def _read_add_column(
        self,
        modifiers: CaseInsensitiveDict[str],
        identifier: str,
        entries: list[tuple[str, str]] | None = None,
    ) -> AddColumn2DA:
        """Loads the add new column to be added to the 2D array.

        Args:
        ----
            modifiers: CaseInsensitiveDict[str]: Dictionary of column modifiers.
            identifier: str: Identifier of the column.

        Returns:
        -------
            AddColumn2DA: Object containing details of added column.

        Processing Logic:
        ----------------
            - Read optional 'ColumnLabel' and 'DefaultValue' values
            - Preserve the original section entries for ordered execution
            - Call column_inserts_2da() to retain the existing parsed fields
            - Return AddColumn2DA object
        """
        header = modifiers.pop("ColumnLabel", "")
        default = modifiers.pop("DefaultValue", "")
        default = default if default != "****" else ""

        index_insert: dict[int, RowValue]
        label_insert: dict[str, RowValue]
        store_2da: dict[int, str]

        index_insert, label_insert, store_2da = self.column_inserts_2da(
            identifier,
            modifiers,
        )
        return AddColumn2DA(
            identifier,
            header,
            default,
            index_insert,
            label_insert,
            store_2da,
            entries,
        )

    def target_2da(
        self,
        identifier: str,
        modifiers: CaseInsensitiveDict[str],
    ) -> Target | None:
        """Gets or creates a 2D target from modifiers.

        Args:
        ----
            identifier: str - Identifier for target
            modifiers: CaseInsensitiveDict[str] - Modifiers dictionary

        Returns:
        -------
            Target | None: Target object or None

        Processing Logic:
        ----------------
            - Checks for RowIndex, RowLabel or LabelIndex key
            - Calls get_target() to create Target object
            - Returns None if no valid key found with warning
        """

        def get_target(
            target_type: TargetType,
            key: str,
            *,
            is_int: bool = False,
        ) -> Target:
            raw_value: str | None = modifiers.pop(key, None)
            if raw_value is None:
                msg = f"[2DAList] parse error: '{key}' missing from [{identifier}] in ini."
                raise ValueError(msg)
            lower_raw_value = raw_value.lower()
            if lower_raw_value.startswith("strref") and len(raw_value) > len("strref") and raw_value[6:].isdigit():
                value: str | int | RowValue2DAMemory | RowValueTLKMemory = RowValueTLKMemory(int(raw_value[6:]))
            elif lower_raw_value.startswith("2damemory") and len(raw_value) > len("2damemory") and raw_value[9:].isdigit():
                value = RowValue2DAMemory(int(raw_value[9:]))
            else:
                value = int(raw_value) if is_int else raw_value
            return Target(target_type, value)

        if "RowIndex" in modifiers:
            return get_target(TargetType.ROW_INDEX, "RowIndex", is_int=True)
        if "RowLabel" in modifiers:
            return get_target(TargetType.ROW_LABEL, "RowLabel")
        if "LabelIndex" in modifiers:
            return get_target(TargetType.LABEL_COLUMN, "LabelIndex")

        self.log.add_warning(f"No line set to be modified in [{identifier}].")
        return None

    def cells_2da(
        self,
        identifier: str,
        modifiers: CaseInsensitiveDict[str],
    ) -> tuple[dict[str, RowValue], dict[int, RowValue], dict[int, RowValue]]:
        """Parses modifiers to extract 2DA and TLK cell values and row labels.

        Args:
        ----
            identifier: str - Section name for this 2DA
            modifiers: CaseInsensitiveDict[str] - Modifiers dictionary

        Returns:
        -------
            tuple[dict[str, RowValue], dict[int, RowValue], dict[int, RowValue]] - tuple containing cells dictionary, 2DA store dictionary, TLK store dictionary

        Processing Logic:
        ----------------
            1. Loops through each modifier and value
            2. Determines modifier type (cell, 2DA store, TLK store, row label)
            3. Creates appropriate RowValue for cell/store value
            4. Adds cell/store value to return dictionaries
        """
        cells: dict[str, RowValue] = {}
        store_2da: dict[int, RowValue] = {}
        store_tlk: dict[int, RowValue] = {}

        for modifier, value in modifiers.items():
            lower_modifier: str = modifier.lower().strip()
            lower_value: str = value.lower()

            is_store_2da: bool = (
                lower_modifier.startswith("2damemory")
                and len(lower_modifier) > len("2damemory")
                and modifier[9:].isdigit()
            )
            is_store_tlk: bool = (
                modifier.startswith("strref")
                and len(lower_modifier) > len("strref")
                and modifier[6:].isdigit()
            )
            is_row_label: bool = lower_modifier in {"rowlabel", "newrowlabel"}

            row_value: RowValue | None = None
            if lower_value.startswith("2damemory"):
                token_id = int(value[9:])
                row_value = RowValue2DAMemory(token_id)
            elif lower_value.startswith("strref"):
                token_id = int(value[6:])
                row_value = RowValueTLKMemory(token_id)
            elif lower_value == "high()":
                row_value = RowValueHigh(None) if modifier == "rowlabel" else RowValueHigh(modifier)
            elif lower_value == "rowindex":
                row_value = RowValueRowIndex()
            elif lower_value == "rowlabel":
                row_value = RowValueRowLabel()
            elif is_store_2da or is_store_tlk:
                row_value = RowValueRowCell(value)
            elif value == "****":
                row_value = RowValueConstant("")
            else:
                row_value = RowValueConstant(value)

            if is_store_2da:
                token_id = int(modifier[9:])
                store_2da[token_id] = row_value
            elif is_store_tlk:
                token_id = int(modifier[6:])
                store_tlk[token_id] = row_value
            elif not is_row_label:
                cells[modifier] = row_value

        return cells, store_2da, store_tlk

    def row_label_2da(
        self,
        identifier: str,
        modifiers: CaseInsensitiveDict[str],
    ) -> RowValue | None:
        """Returns the row label for a 2D array based on modifiers.

        Args:
        ----
            identifier: Identifier for the 2D array
            modifiers: CaseInsensitiveDict of modifiers for the 2D array

        Returns:
        -------
            str | None: The row label or None if not found

        Processing Logic:
        ----------------
            - Check if 'RowLabel' exists as a key in modifiers
            - If not present, check if 'NewRowLabel' exists as a key
            - Return the value of the key if present, else return None.
        """
        raw_value = modifiers.pop(
            "RowLabel",
            modifiers.pop("NewRowLabel", None),
        )
        if raw_value is None:
            return None

        lower_value = raw_value.lower().strip()
        if lower_value == "high()":
            return RowValueHigh(None)
        if lower_value.startswith("2damemory") and raw_value[9:].isdigit():
            return RowValue2DAMemory(int(raw_value[9:]))
        if lower_value.startswith("strref") and raw_value[6:].isdigit():
            return RowValueTLKMemory(int(raw_value[6:]))
        return RowValueConstant(raw_value)

    def column_inserts_2da(
        self,
        identifier: str,
        modifiers: CaseInsensitiveDict[str],
    ) -> tuple[dict[int, RowValue], dict[str, RowValue], dict[int, str]]:
        """Extracts specific 2DA patch information from the ini.

        Args:
        ----
            identifier: str - Section name being handled
            modifiers: CaseInsensitiveDict[str] - Modifiers to insert values

        Returns:
        -------
            tuple[dict[int, RowValue], dict[str, RowValue], dict[int, str]] - Index inserts, label inserts, 2DA store

        Processes Logic:
        ---------------
            - Loops through modifiers and extracts value type
            - Assigns row value based on value type
            - Inserts into appropriate return dictionary based on modifier key
            - Returns tuple of inserted values dictionaries.
        """
        index_insert: dict[int, RowValue] = {}
        label_insert: dict[str, RowValue] = {}
        store_2da: dict[int, str] = {}

        for modifier, value in modifiers.items():
            modifier_lowercase: str = modifier.lower()
            value_lowercase: str = value.lower()

            is_store_2da: bool = value_lowercase.startswith("2damemory")
            is_store_tlk: bool = value_lowercase.startswith("strref")

            row_value: RowValue | None = None
            if is_store_2da:
                token_id = int(value[9:])
                row_value = RowValue2DAMemory(token_id)
            elif is_store_tlk:
                token_id = int(value[6:])
                row_value = RowValueTLKMemory(token_id)
            else:
                row_value = RowValueConstant(value)

            if modifier_lowercase.startswith("i") and modifier[1:].isdigit():
                index = int(modifier[1:])
                index_insert[index] = row_value
            elif modifier_lowercase.startswith("l"):
                label: str = modifier[1:]
                label_insert[label] = row_value
            elif modifier_lowercase.startswith("2damemory"):
                token_id = int(modifier[9:])
                store_2da[token_id] = value

        return index_insert, label_insert, store_2da

    #################

    @staticmethod
    def normalize_tslpatcher_float(value_str: str) -> str:
        """Normalize a float value string by replacing commas with periods.

        Args:
        ----
            value_str: String value to normalize

        Returns:
        -------
            str: Normalized string value with commas replaced with periods
        """
        return value_str.replace(",", ".")

    @staticmethod
    def normalize_tslpatcher_crlf(value_str: str) -> str:
        r"""Normalize line endings in a string value.

        Args:
        ----
            value_str: String value to normalize line endings

        Returns:
        -------
            str: String with normalized line endings

        Processes line endings:
            - Replaces "<#LF#>" with "\n"
            - Replaces "<#CR#>" with "\r"
            - Returns string with all line endings normalized
        """
        return value_str.replace("<#LF#>", "\n").replace("<#CR#>", "\r")

    @staticmethod
    def resolve_tslpatcher_ssf_sound(name: str) -> SSFSound:
        """Resolves a config string to an SSFSound enum value.

        Args:
        ----
            name (str): The config string name

        Returns:
        -------
            SSFSound: The resolved SSFSound enum value

        Processing Logic:
        ----------------
            - Defines a CaseInsensitiveDict mapping config strings to SSFSound enum values
            - Looks up the provided name in the dict and returns the corresponding SSFSound value.
        """
        configstr_to_ssfsound = CaseInsensitiveDict(
            {
                "Battlecry 1": SSFSound.BATTLE_CRY_1,
                "Battlecry 2": SSFSound.BATTLE_CRY_2,
                "Battlecry 3": SSFSound.BATTLE_CRY_3,
                "Battlecry 4": SSFSound.BATTLE_CRY_4,
                "Battlecry 5": SSFSound.BATTLE_CRY_5,
                "Battlecry 6": SSFSound.BATTLE_CRY_6,
                "Selected 1": SSFSound.SELECT_1,
                "Selected 2": SSFSound.SELECT_2,
                "Selected 3": SSFSound.SELECT_3,
                "Attack 1": SSFSound.ATTACK_GRUNT_1,
                "Attack 2": SSFSound.ATTACK_GRUNT_2,
                "Attack 3": SSFSound.ATTACK_GRUNT_3,
                "Pain 1": SSFSound.PAIN_GRUNT_1,
                "Pain 2": SSFSound.PAIN_GRUNT_2,
                "Low health": SSFSound.LOW_HEALTH,
                "Death": SSFSound.DEAD,
                "Critical hit": SSFSound.CRITICAL_HIT,
                "Target immune": SSFSound.TARGET_IMMUNE,
                "Place mine": SSFSound.LAY_MINE,
                "Disarm mine": SSFSound.DISARM_MINE,
                "Stealth on": SSFSound.BEGIN_STEALTH,
                "Search": SSFSound.BEGIN_SEARCH,
                "Pick lock start": SSFSound.BEGIN_UNLOCK,
                "Pick lock fail": SSFSound.UNLOCK_FAILED,
                "Pick lock done": SSFSound.UNLOCK_SUCCESS,
                "Leave party": SSFSound.SEPARATED_FROM_PARTY,
                "Rejoin party": SSFSound.REJOINED_PARTY,
                "Poisoned": SSFSound.POISONED,
                "Unknown(29)": SSFSound.UNKNOWN_29,
                "Unknown(30)": SSFSound.UNKNOWN_30,
                "Unknown(31)": SSFSound.UNKNOWN_31,
                "Unknown(32)": SSFSound.UNKNOWN_32,
                "Unknown(33)": SSFSound.UNKNOWN_33,
                "Unknown(34)": SSFSound.UNKNOWN_34,
                "Unknown(35)": SSFSound.UNKNOWN_35,
                "Unknown(36)": SSFSound.UNKNOWN_36,
                "Unknown(37)": SSFSound.UNKNOWN_37,
                "Unknown(38)": SSFSound.UNKNOWN_38,
                "Unknown(39)": SSFSound.UNKNOWN_39,
                "Unknown(40)": SSFSound.UNKNOWN_40,
            }.items(),
        )
        return configstr_to_ssfsound[name]

    @staticmethod
    def resolve_tslpatcher_gff_field_type(field_type_num_str: str) -> GFFFieldType:
        """Resolves a TSLPatcher GFF field type to a PyKotor GFFFieldType enum.

        Use this function to work with the ini's FieldType= values in PyKotor.

        Args:
        ----
            field_type_num_str: {String containing the field type number}.

        Returns:
        -------
            GFFFieldType: {The GFFFieldType enum value corresponding to the input string}

        Processing Logic:
        ----------------
            - Defines a dictionary mapping field type number strings to GFFFieldType enum values
            - Looks up the input string in the dictionary
            - Returns the corresponding GFFFieldType value.
        """
        fieldname_to_fieldtype = CaseInsensitiveDict(
            {
                "Binary": GFFFieldType.Binary,  # HoloPatcher only.
                "Byte": GFFFieldType.UInt8,
                "Char": GFFFieldType.Int8,
                "Word": GFFFieldType.UInt16,
                "Short": GFFFieldType.Int16,
                "DWORD": GFFFieldType.UInt32,
                "Int": GFFFieldType.Int32,
                "Int64": GFFFieldType.Int64,
                "Float": GFFFieldType.Single,
                "Double": GFFFieldType.Double,
                "ExoString": GFFFieldType.String,
                "ResRef": GFFFieldType.ResRef,
                "ExoLocString": GFFFieldType.LocalizedString,
                "Position": GFFFieldType.Vector3,
                "Orientation": GFFFieldType.Vector4,
                "Struct": GFFFieldType.Struct,
                "List": GFFFieldType.List,
            }.items(),
        )
        return fieldname_to_fieldtype[field_type_num_str]
