from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from pykotor.resource.generics._gff import (
    GFFInteger,
    bind_gff_struct,
    preserve_gff,
    remember_gff,
    remember_gff_struct,
    update_gff_membership,
)
from pykotor.common.language import LocalizedString
from pykotor.common.misc import EquipmentSlot, Game, InventoryItem, ResRef
from pykotor.resource.formats.gff import GFF, GFFContent, GFFList, read_gff, write_gff
from pykotor.resource.formats.gff.gff_auto import bytes_gff
from pykotor.resource.formats.gff.gff_data import GFFFieldType
from pykotor.resource.type import ResourceType
from utility.logger_util import RobustRootLogger

if TYPE_CHECKING:
    from pykotor.resource.type import SOURCE_TYPES, TARGET_TYPES


class UTC:
    """Stores creature data.

    Attributes:
    ----------
        resref: "TemplateResRef" field.
        tag: "Tag" field.
        comment: "Comment" field.
        conversation: "Conversation" field.
        first_name: "FirstName" field.
        last_name: "LastName" field.
        subrace_id: "SubraceIndex" field.
        perception_id: "PerceptionRange" field.
        race_id: "Race" field.
        appearance_id: "Appearance_Type" field.
        gender_id: "Gender" field.
        faction_id: "FactionID" field.
        walkrate_id: "WalkRate" field.
        soundset_id: "SoundSetFile" field.
        portrait_id: "PortraitId" field.
        body_variation: "BodyVariation" field.
        texture_variation: "TextureVar" field.
        not_reorienting: "NotReorienting" field.
        party_interact: "PartyInteract" field.
        no_perm_death: "NoPermDeath" field.
        min1_hp: "Min1HP" field.
        plot: "Plot" field.
        interruptable: "Interruptable" field.
        is_pc: "IsPC" field.
        disarmable: "Disarmable" field.
        alignment: "GoodEvil" field.
        challenge_rating: "ChallengeRating" field.
        blindspot: "BlindSpot" field.
        multiplier_set: "MultiplierSet" field.
        natural_ac: "NaturalAC" field.
        reflex_bonus: "refbonus" field.
        willpower_bonus: "willbonus" field.
        fortitude_bonus: "fortbonus" field.
        strength: "Str" field.
        dexterity: "Dex" field.
        constitution: "Con" field.
        intelligence: "Int" field.
        wisdom: "Wis" field.
        charisma: "Cha" field.
        current_hp: "CurrentHitPoints" field.
        max_hp: "MaxHitPoints" field.
        hp: "HitPoints" field.
        fp: "CurrentForce" field.
        max_fp: "ForcePoints" field.
        on_end_dialog: "ScriptEndDialogu" field.
        on_blocked: "ScriptOnBlocked" field.
        on_heartbeat: "ScriptHeartbeat" field.
        on_notice: "ScriptOnNotice" field.
        on_spell: "ScriptSpellAt" field.
        on_attacked: "ScriptAttacked" field.
        on_damaged: "ScriptDamaged" field.
        on_disturbed: "ScriptDisturbed" field.
        on_end_round: "ScriptEndRound" field.
        on_dialog: "ScriptDialogue" field.
        on_spawn: "ScriptSpawn" field.
        on_rested: "ScriptRested" field.
        on_death: "ScriptDeath" field.
        on_user_defined: "ScriptUserDefine" field.

        ignore_cre_path: "IgnoreCrePath" field. KotOR 2 Only.
        hologram: "Hologram" field. KotOR 2 Only.

        palette_id: "PaletteID" field. Used in toolset only.

        bodybag_id: "BodyBag" field. Not used by the game engine.
        deity: "Deity" field. Not used by the game engine.
        description: "Description" field. Not used by the game engine.
        lawfulness: "LawfulChaotic" field. Not used by the game engine.
        phenotype_id: "Phenotype" field. Not used by the game engine.
        on_rested: "ScriptRested" field. Not used by the game engine.
        subrace_name: "Subrace" field. Not used by the game engine.
    """

    BINARY_TYPE = ResourceType.UTC

    def __init__(
        self,
    ):
        self._extra_unimplemented_skills: list[int] = []

        self.resref: ResRef = ResRef.from_blank()
        self.conversation: ResRef = ResRef.from_blank()
        self.tag: str = ""
        self.comment: str = ""

        self.first_name: LocalizedString = LocalizedString.from_invalid()
        self.last_name: LocalizedString = LocalizedString.from_invalid()

        self.subrace_id: int = 0
        self.portrait_id: int = 0
        self.perception_id: int = 0
        self.race_id: int = 0
        self.appearance_id: int = 0
        self.gender_id: int = 0
        self.faction_id: int = 0
        self.walkrate_id: int = 0
        self.soundset_id: int = 0
        self.save_will: int = 0
        self.save_fortitude: int = 0
        self.morale: int = 0
        self.morale_recovery: int = 0
        self.morale_breakpoint: int = 0

        self.body_variation: int = 0
        self.texture_variation: int = 0

        self.portrait_resref: ResRef = ResRef.from_blank()

        self.not_reorienting: bool = False
        self.party_interact: bool = False
        self.no_perm_death: bool = False
        self.min1_hp: bool = False
        self.plot: bool = False
        self.interruptable: bool = False
        self.is_pc: bool = False  # ???
        self.disarmable: bool = False  # ???
        self.ignore_cre_path: bool = False  # KotOR 2 Only
        self.hologram: bool = False  # KotOR 2 Only
        self.will_not_render: bool = False  # Kotor 2 Only

        self.alignment: int = 0
        self.challenge_rating: float = 0.0
        self.blindspot: float = 0.0  # KotOR 2 Only
        self.multiplier_set: int = 0  # KotOR 2 Only

        self.natural_ac: int = 0
        self.reflex_bonus: int = 0
        self.willpower_bonus: int = 0
        self.fortitude_bonus: int = 0

        self.current_hp: int = 0
        self.max_hp: int = 0
        self.hp: int = 0

        self.max_fp: int = 0
        self.fp: int = 0

        self.strength: int = 0
        self.dexterity: int = 0
        self.constitution: int = 0
        self.intelligence: int = 0
        self.wisdom: int = 0
        self.charisma: int = 0

        self.computer_use: int = 0
        self.demolitions: int = 0
        self.stealth: int = 0
        self.awareness: int = 0
        self.persuade: int = 0
        self.repair: int = 0
        self.security: int = 0
        self.treat_injury: int = 0

        self.on_end_dialog: ResRef = ResRef.from_blank()
        self.on_blocked: ResRef = ResRef.from_blank()
        self.on_heartbeat: ResRef = ResRef.from_blank()
        self.on_notice: ResRef = ResRef.from_blank()
        self.on_spell: ResRef = ResRef.from_blank()
        self.on_attacked: ResRef = ResRef.from_blank()
        self.on_damaged: ResRef = ResRef.from_blank()
        self.on_disturbed: ResRef = ResRef.from_blank()
        self.on_end_round: ResRef = ResRef.from_blank()
        self.on_dialog: ResRef = ResRef.from_blank()
        self.on_spawn: ResRef = ResRef.from_blank()
        self.on_rested: ResRef = ResRef.from_blank()
        self.on_death: ResRef = ResRef.from_blank()
        self.on_user_defined: ResRef = ResRef.from_blank()

        self.classes: list[UTCClass] = []
        self.special_abilities: list[UTCSpecialAbility] = []
        self.feats: list[int] = []
        self.inventory: list[InventoryItem] = []
        self.equipment: dict[EquipmentSlot, InventoryItem] = {}
        self._equipment_rows: list[tuple[EquipmentSlot, InventoryItem]] = []

        # Deprecated:
        self.palette_id: int = 0
        self.bodybag_id: int = 1
        self.deity: str = ""
        self.description: LocalizedString = LocalizedString.from_invalid()
        self.lawfulness: int = 0
        self.phenotype_id: int = 0
        # self.on_rested: ResRef = ResRef.from_blank()
        self.subrace_name: str = ""

    def update_feat_membership(self, selected: Iterable[int]) -> None:
        """Retain selected feat records and fill removed slots with new feats.

        Existing duplicates remain separate records. Assign or reorder ``feats``
        directly when the requested operation is positional rather than a
        membership selection.
        """
        self.feats[:] = update_gff_membership(
            [self.feats], selected, key=int, create=int,
        )[0]

    def update_power_membership(self, selected: Iterable[int]) -> None:
        """Update the shared power selection without moving retained class records.

        New powers fill vacancies in class/list order; any remainder is appended
        to the last class. At least one class is required to add a power.
        """
        groups = update_gff_membership(
            [utc_class.powers for utc_class in self.classes], selected, key=int, create=int,
        )
        for utc_class, powers in zip(self.classes, groups):
            utc_class.powers[:] = powers


class UTCSpecialAbility:
    """An entry in the engine's SpecAbilityList."""

    def __init__(self, spell: int = 0, flags: int = 0, caster_level: int = 0):
        self.spell = spell
        self.flags = flags
        self.caster_level = caster_level


class UTCClass:
    def __init__(
        self,
        class_id: int,
        class_level: int = 0,
    ):
        self.class_id: int = class_id
        self.class_level: int = class_level
        self.powers: list[int] = []

    def __repr__(
        self,
    ):
        return f"{self.__class__.__name__}(class_id={self.class_id}, class_level={self.class_level})"

    def __eq__(
        self,
        other: UTCClass | object,
    ):
        if self is other:
            return True
        if isinstance(other, UTCClass):
            return self.class_id == other.class_id and self.class_level == self.class_level
        return NotImplemented


def construct_utc(
    gff: GFF,
) -> UTC:
    utc = UTC()

    root = gff.root
    utc.resref = root.acquire("TemplateResRef", ResRef.from_blank(), field_type=GFFFieldType.ResRef)
    utc.tag = root.acquire("Tag", "", str, field_type=GFFFieldType.String)
    utc.comment = root.acquire("Comment", "", str, field_type=GFFFieldType.String)
    utc.conversation = root.acquire("Conversation", ResRef.from_blank(), field_type=GFFFieldType.ResRef)

    utc.first_name = root.acquire("FirstName", LocalizedString.from_invalid(), field_type=GFFFieldType.LocalizedString)
    utc.last_name = root.acquire("LastName", LocalizedString.from_invalid(), field_type=GFFFieldType.LocalizedString)

    utc.subrace_id = root.acquire("SubraceIndex", 0, field_type=GFFFieldType.UInt8)
    utc.perception_id = root.acquire("PerceptionRange", 0, field_type=GFFFieldType.UInt8)
    utc.race_id = root.acquire("Race", 0, field_type=GFFFieldType.UInt8)
    utc.appearance_id = root.acquire("Appearance_Type", 0, field_type=GFFFieldType.UInt16)
    utc.gender_id = root.acquire("Gender", 0, field_type=GFFFieldType.UInt8)
    utc.faction_id = root.acquire("FactionID", 0, field_type=GFFFieldType.UInt16)
    utc.walkrate_id = root.acquire("WalkRate", 0, field_type=GFFFieldType.Int32)
    utc.soundset_id = root.acquire("SoundSetFile", 0, field_type=GFFFieldType.UInt16)
    utc.portrait_id = root.acquire("PortraitId", 0, field_type=GFFFieldType.UInt16)
    utc.palette_id = root.acquire("PaletteID", 0, field_type=GFFFieldType.UInt8)
    utc.bodybag_id = root.acquire("BodyBag", 0, field_type=GFFFieldType.UInt8)

    # TODO(th3w1zard1): Add these seemingly missing fields into UTCEditor?
    utc.portrait_resref = root.acquire("Portrait", ResRef.from_blank(), field_type=GFFFieldType.ResRef)
    utc.save_will = root.acquire("SaveWill", 0, field_type=GFFFieldType.UInt8)
    utc.save_fortitude = root.acquire("SaveFortitude", 0, field_type=GFFFieldType.UInt8)
    utc.morale = root.acquire("Morale", 0, field_type=GFFFieldType.UInt8)
    utc.morale_recovery = root.acquire("MoraleRecovery", 0, field_type=GFFFieldType.UInt8)
    utc.morale_breakpoint = root.acquire("MoraleBreakpoint", 0, field_type=GFFFieldType.UInt8)

    utc.body_variation = root.acquire("BodyVariation", 0, field_type=GFFFieldType.UInt8)
    utc.texture_variation = root.acquire("TextureVar", 0, field_type=GFFFieldType.UInt8)

    utc.not_reorienting = bool(root.acquire("NotReorienting", 0, field_type=GFFFieldType.UInt8))
    utc.party_interact = bool(root.acquire("PartyInteract", 0, field_type=GFFFieldType.UInt8))
    utc.no_perm_death = bool(root.acquire("NoPermDeath", 0, field_type=GFFFieldType.UInt8))
    utc.min1_hp = bool(root.acquire("Min1HP", 0, field_type=GFFFieldType.UInt8))
    utc.plot = bool(root.acquire("Plot", 0, field_type=GFFFieldType.UInt8))
    utc.interruptable = bool(root.acquire("Interruptable", 0, field_type=GFFFieldType.UInt8))
    utc.is_pc = bool(root.acquire("IsPC", 0, field_type=GFFFieldType.UInt8))
    utc.disarmable = bool(root.acquire("Disarmable", 0, field_type=GFFFieldType.UInt8))
    utc.ignore_cre_path = bool(root.acquire("IgnoreCrePath", 0, field_type=GFFFieldType.UInt8))
    utc.hologram = bool(root.acquire("Hologram", 0, field_type=GFFFieldType.UInt8))
    utc.will_not_render = bool(root.acquire("WillNotRender", 0, field_type=GFFFieldType.UInt8))

    utc.alignment = root.acquire("GoodEvil", 0, field_type=GFFFieldType.UInt8)
    utc.challenge_rating = root.acquire("ChallengeRating", 0.0, field_type=GFFFieldType.Single)
    utc.blindspot = root.acquire("BlindSpot", 0.0, field_type=GFFFieldType.Single)
    utc.multiplier_set = root.acquire("MultiplierSet", 0, field_type=GFFFieldType.UInt8)

    utc.natural_ac = root.acquire("NaturalAC", 0, field_type=GFFFieldType.UInt8)
    utc.reflex_bonus = root.acquire("refbonus", 0, field_type=GFFFieldType.Int16)
    utc.willpower_bonus = root.acquire("willbonus", 0, field_type=GFFFieldType.Int16)
    utc.fortitude_bonus = root.acquire("fortbonus", 0, field_type=GFFFieldType.Int16)

    utc.strength = root.acquire("Str", 0, field_type=GFFFieldType.UInt8)
    utc.dexterity = root.acquire("Dex", 0, field_type=GFFFieldType.UInt8)
    utc.constitution = root.acquire("Con", 0, field_type=GFFFieldType.UInt8)
    utc.intelligence = root.acquire("Int", 0, field_type=GFFFieldType.UInt8)
    utc.wisdom = root.acquire("Wis", 0, field_type=GFFFieldType.UInt8)
    utc.charisma = root.acquire("Cha", 0, field_type=GFFFieldType.UInt8)

    utc.current_hp = root.acquire("CurrentHitPoints", 0, field_type=GFFFieldType.Int16)
    utc.max_hp = root.acquire("MaxHitPoints", 0, field_type=GFFFieldType.Int16)
    utc.hp = root.acquire("HitPoints", 0, field_type=GFFFieldType.Int16)
    utc.max_fp = root.acquire("ForcePoints", 0, field_type=GFFFieldType.Int16)
    utc.fp = root.acquire("CurrentForce", 0, field_type=GFFFieldType.Int16)

    utc.on_end_dialog = root.acquire("ScriptEndDialogu", ResRef.from_blank(), field_type=GFFFieldType.ResRef)
    utc.on_blocked = root.acquire("ScriptOnBlocked", ResRef.from_blank(), field_type=GFFFieldType.ResRef)
    utc.on_heartbeat = root.acquire("ScriptHeartbeat", ResRef.from_blank(), field_type=GFFFieldType.ResRef)
    utc.on_notice = root.acquire("ScriptOnNotice", ResRef.from_blank(), field_type=GFFFieldType.ResRef)
    utc.on_spell = root.acquire("ScriptSpellAt", ResRef.from_blank(), field_type=GFFFieldType.ResRef)
    utc.on_attacked = root.acquire("ScriptAttacked", ResRef.from_blank(), field_type=GFFFieldType.ResRef)
    utc.on_damaged = root.acquire("ScriptDamaged", ResRef.from_blank(), field_type=GFFFieldType.ResRef)
    utc.on_disturbed = root.acquire("ScriptDisturbed", ResRef.from_blank(), field_type=GFFFieldType.ResRef)
    utc.on_end_round = root.acquire("ScriptEndRound", ResRef.from_blank(), field_type=GFFFieldType.ResRef)
    utc.on_dialog = root.acquire("ScriptDialogue", ResRef.from_blank(), field_type=GFFFieldType.ResRef)
    utc.on_spawn = root.acquire("ScriptSpawn", ResRef.from_blank(), field_type=GFFFieldType.ResRef)
    utc.on_rested = root.acquire("ScriptRested", ResRef.from_blank(), field_type=GFFFieldType.ResRef)
    utc.on_death = root.acquire("ScriptDeath", ResRef.from_blank(), field_type=GFFFieldType.ResRef)
    utc.on_user_defined = root.acquire("ScriptUserDefine", ResRef.from_blank(), field_type=GFFFieldType.ResRef)

    skill_list = root.acquire("SkillList", GFFList(), field_type=GFFFieldType.List)
    skill_names = ("computer_use", "demolitions", "stealth", "awareness", "persuade", "repair", "security", "treat_injury")
    for index, name in enumerate(skill_names):
        skill_struct = skill_list.at(index)
        setattr(utc, name, 0 if skill_struct is None else skill_struct.acquire("Rank", 0, field_type=GFFFieldType.UInt8))

    # Not sure why there's extras... some utc's in k1 have 20 structs in the SkillList.
    if len(skill_list._structs) > 8:
        utc._extra_unimplemented_skills = [skill_struct.acquire("Rank", 0, field_type=GFFFieldType.UInt8) for skill_struct in skill_list._structs[8:]]

    class_list: GFFList = root.acquire("ClassList", GFFList(), field_type=GFFFieldType.List)
    for class_struct in class_list:
        class_id = class_struct.acquire("Class", 0, field_type=GFFFieldType.Int32)
        class_level = class_struct.acquire("ClassLevel", 0, field_type=GFFFieldType.Int16)
        utc_class = UTCClass(class_id, class_level)
        remember_gff_struct(utc_class, class_struct)

        power_list: GFFList = class_struct.acquire("KnownList0", GFFList(), field_type=GFFFieldType.List)
        for power_struct in power_list:
            spell_thing = GFFInteger(power_struct.acquire("Spell", 0, field_type=GFFFieldType.UInt16))
            remember_gff_struct(spell_thing, power_struct)
            utc_class.powers.append(spell_thing)

        utc.classes.append(utc_class)

    feat_list: GFFList = root.acquire("FeatList", GFFList(), field_type=GFFFieldType.List)
    for feat_struct in feat_list:
        feat_id_thing = GFFInteger(feat_struct.acquire("Feat", 0, field_type=GFFFieldType.UInt16))
        remember_gff_struct(feat_id_thing, feat_struct)
        utc.feats.append(feat_id_thing)

    equipment_list: GFFList = root.acquire("Equip_ItemList", GFFList(), field_type=GFFFieldType.List)
    for equipment_struct in equipment_list:
        slot = EquipmentSlot(equipment_struct.struct_id)
        resref = equipment_struct.acquire("EquippedRes", ResRef.from_blank(), field_type=GFFFieldType.ResRef)
        droppable = bool(equipment_struct.acquire("Dropable", 0, field_type=GFFFieldType.UInt8))
        utc.equipment[slot] = InventoryItem(resref, droppable)
        remember_gff_struct(utc.equipment[slot], equipment_struct)
        utc._equipment_rows.append((slot, utc.equipment[slot]))

    item_list: GFFList = root.acquire("ItemList", GFFList(), field_type=GFFFieldType.List)
    for item_struct in item_list:
        resref = item_struct.acquire("InventoryRes", ResRef.from_blank(), field_type=GFFFieldType.ResRef)
        droppable = bool(item_struct.acquire("Dropable", 0, field_type=GFFFieldType.UInt8))
        item = InventoryItem(resref, droppable)
        remember_gff_struct(item, item_struct)
        item.pos_x = item_struct.acquire("Repos_PosX", 0)
        item.pos_y = item_struct.acquire("Repos_Posy", 0)
        utc.inventory.append(item)

    for ability_struct in root.acquire("SpecAbilityList", GFFList(), field_type=GFFFieldType.List):
        ability = UTCSpecialAbility(
            ability_struct.acquire("Spell", 0, field_type=GFFFieldType.UInt16),
            ability_struct.acquire("SpellFlags", 0, field_type=GFFFieldType.UInt8),
            ability_struct.acquire("SpellCasterLevel", 0, field_type=GFFFieldType.UInt8),
        )
        remember_gff_struct(ability, ability_struct)
        utc.special_abilities.append(ability)

    remember_gff(utc, gff)
    return utc


def dismantle_utc(
    utc: UTC,
    game: Game = Game.K2,
    *,
    use_deprecated: bool = True,
) -> GFF:
    gff = GFF(GFFContent.UTC)

    root = gff.root
    root.set_resref("TemplateResRef", utc.resref)
    root.set_string("Tag", utc.tag)
    root.set_string("Comment", utc.comment)
    root.set_resref("Conversation", utc.conversation)

    root.set_locstring("FirstName", utc.first_name)
    root.set_locstring("LastName", utc.last_name)

    root.set_uint8("SubraceIndex", utc.subrace_id)
    root.set_uint8("PerceptionRange", utc.perception_id)
    root.set_uint8("Race", utc.race_id)
    root.set_uint16("Appearance_Type", utc.appearance_id)
    root.set_uint8("Gender", utc.gender_id)
    root.set_uint16("FactionID", utc.faction_id)
    root.set_int32("WalkRate", utc.walkrate_id)
    root.set_uint16("SoundSetFile", utc.soundset_id)
    root.set_uint16("PortraitId", utc.portrait_id)

    # TODO(th3w1zard1): Add these seemingly missing fields into UTCEditor?
    root.set_resref("Portrait", utc.portrait_resref)
    root.set_uint8("SaveWill", utc.save_will)
    root.set_uint8("SaveFortitude", utc.save_fortitude)
    root.set_uint8("Morale", utc.morale)
    root.set_uint8("MoraleRecovery", utc.morale_recovery)
    root.set_uint8("MoraleBreakpoint", utc.morale_breakpoint)

    root.set_uint8("BodyVariation", utc.body_variation)
    root.set_uint8("TextureVar", utc.texture_variation)

    root.set_uint8("NotReorienting", utc.not_reorienting)
    root.set_uint8("PartyInteract", utc.party_interact)
    root.set_uint8("NoPermDeath", utc.no_perm_death)
    root.set_uint8("Min1HP", utc.min1_hp)
    root.set_uint8("Plot", utc.plot)
    root.set_uint8("Interruptable", utc.interruptable)
    root.set_uint8("IsPC", utc.is_pc)
    root.set_uint8("Disarmable", utc.disarmable)

    root.set_uint8("GoodEvil", utc.alignment)
    root.set_single("ChallengeRating", utc.challenge_rating)

    root.set_uint8("NaturalAC", utc.natural_ac)
    root.set_int16("refbonus", utc.reflex_bonus)
    root.set_int16("willbonus", utc.willpower_bonus)
    root.set_int16("fortbonus", utc.fortitude_bonus)

    root.set_uint8("Str", utc.strength)
    root.set_uint8("Dex", utc.dexterity)
    root.set_uint8("Con", utc.constitution)
    root.set_uint8("Int", utc.intelligence)
    root.set_uint8("Wis", utc.wisdom)
    root.set_uint8("Cha", utc.charisma)

    root.set_int16("CurrentHitPoints", utc.current_hp)
    root.set_int16("MaxHitPoints", utc.max_hp)
    root.set_int16("HitPoints", utc.hp)
    root.set_int16("CurrentForce", utc.fp)
    root.set_int16("ForcePoints", utc.max_fp)

    root.set_resref("ScriptEndDialogu", utc.on_end_dialog)
    root.set_resref("ScriptOnBlocked", utc.on_blocked)
    root.set_resref("ScriptHeartbeat", utc.on_heartbeat)
    root.set_resref("ScriptOnNotice", utc.on_notice)
    root.set_resref("ScriptSpellAt", utc.on_spell)
    root.set_resref("ScriptAttacked", utc.on_attacked)
    root.set_resref("ScriptDamaged", utc.on_damaged)
    root.set_resref("ScriptDisturbed", utc.on_disturbed)
    root.set_resref("ScriptEndRound", utc.on_end_round)
    root.set_resref("ScriptDialogue", utc.on_dialog)
    root.set_resref("ScriptSpawn", utc.on_spawn)
    root.set_resref("ScriptRested", utc.on_rested)
    root.set_resref("ScriptDeath", utc.on_death)
    root.set_resref("ScriptUserDefine", utc.on_user_defined)

    root.set_uint8("PaletteID", utc.palette_id)

    skill_list: GFFList = root.set_list("SkillList", GFFList())
    skill_list.add(0).set_uint8("Rank", utc.computer_use)
    skill_list.add(0).set_uint8("Rank", utc.demolitions)
    skill_list.add(0).set_uint8("Rank", utc.stealth)
    skill_list.add(0).set_uint8("Rank", utc.awareness)
    skill_list.add(0).set_uint8("Rank", utc.persuade)
    skill_list.add(0).set_uint8("Rank", utc.repair)
    skill_list.add(0).set_uint8("Rank", utc.security)
    skill_list.add(0).set_uint8("Rank", utc.treat_injury)

    class_list: GFFList = root.set_list("ClassList", GFFList())
    for utc_class in utc.classes:
        class_struct = class_list.add(2)
        bind_gff_struct(class_struct, utc_class)
        class_struct.set_int32("Class", utc_class.class_id)
        class_struct.set_int16("ClassLevel", utc_class.class_level)
        power_list: GFFList = class_struct.set_list("KnownList0", GFFList())
        for power in utc_class.powers:
            power_struct = power_list.add(3)
            bind_gff_struct(power_struct, power)
            power_struct.set_uint16("Spell", power)
            power_struct.set_uint8("SpellFlags", 1)
            power_struct.set_uint8("SpellMetaMagic", 0)


    ability_list = root.set_list("SpecAbilityList", GFFList())
    for ability in utc.special_abilities:
        ability_struct = ability_list.add(0)
        bind_gff_struct(ability_struct, ability)
        ability_struct.set_uint16("Spell", ability.spell)
        ability_struct.set_uint8("SpellFlags", ability.flags)
        ability_struct.set_uint8("SpellCasterLevel", ability.caster_level)

    feat_list: GFFList = root.set_list("FeatList", GFFList())
    for feat in utc.feats:
        feat_struct = feat_list.add(1)
        bind_gff_struct(feat_struct, feat)
        feat_struct.set_uint16("Feat", feat)

    # Not sure what these are for, verified they exist in K1's 'c_drdg.utc' in data\templates.bif. Might be unused in which case this can be deleted.
    if utc._extra_unimplemented_skills:
        for val in utc._extra_unimplemented_skills:
            skill_list.add(0).set_uint8("Rank", val)

    equipment_list: GFFList = root.set_list("Equip_ItemList", GFFList())
    last_equipment_rows = {slot: i for i, (slot, _) in enumerate(utc._equipment_rows)}
    equipment_rows = [
        (slot, utc.equipment[slot] if i == last_equipment_rows[slot] else item)
        for i, (slot, item) in enumerate(utc._equipment_rows) if slot in utc.equipment
    ]
    equipment_rows.extend((slot, item) for slot, item in utc.equipment.items() if slot not in last_equipment_rows)
    for slot, item in equipment_rows:
        equipment_struct = equipment_list.add(slot.value)
        bind_gff_struct(equipment_struct, item, preserve_id=False)
        equipment_struct.set_resref("EquippedRes", item.resref)
        if item.droppable:
            equipment_struct.set_uint8("Dropable", value=True)

    item_list: GFFList = root.set_list("ItemList", GFFList())
    for i, item in enumerate(utc.inventory):
        item_struct = item_list.add(i)
        bind_gff_struct(item_struct, item)
        item_struct.set_resref("InventoryRes", item.resref)
        item_struct.set_uint16("Repos_PosX", i if item.pos_x is None else item.pos_x)
        item_struct.set_uint16("Repos_Posy", 0 if item.pos_y is None else item.pos_y)
        if item.droppable:
            item_struct.set_uint8("Dropable", value=True)

    if game.is_k2():
        root.set_single("BlindSpot", utc.blindspot)
        root.set_uint8("MultiplierSet", utc.multiplier_set)
        root.set_uint8("IgnoreCrePath", utc.ignore_cre_path)
        root.set_uint8("Hologram", utc.hologram)
        root.set_uint8("WillNotRender", utc.will_not_render)

    if use_deprecated:
        root.set_uint8("BodyBag", utc.bodybag_id)
        root.set_string("Deity", utc.deity)
        root.set_locstring("Description", utc.description)
        root.set_uint8("LawfulChaotic", utc.lawfulness)
        root.set_int32("Phenotype", utc.phenotype_id)
        root.set_resref("ScriptRested", utc.on_rested)
        root.set_string("Subrace", utc.subrace_name)
        root.set_list("TemplateList", GFFList())

    return preserve_gff(utc, gff, lambda original: dismantle_utc(original, game=game, use_deprecated=use_deprecated))


def read_utc(
    source: SOURCE_TYPES,
    offset: int = 0,
    size: int | None = None,
) -> UTC:
    gff: GFF = read_gff(source, offset, size)
    return construct_utc(gff)


def write_utc(
    utc: UTC,
    target: TARGET_TYPES,
    game: Game = Game.K2,
    file_format: ResourceType = ResourceType.GFF,
    *,
    use_deprecated: bool = True,
):
    gff: GFF = dismantle_utc(utc, game, use_deprecated=use_deprecated)
    write_gff(gff, target, file_format)


def bytes_utc(
    utc: UTC,
    game: Game = Game.K2,
    file_format: ResourceType = ResourceType.GFF,
    *,
    use_deprecated: bool = True,
) -> bytes:
    gff: GFF = dismantle_utc(utc, game, use_deprecated=use_deprecated)
    return bytes_gff(gff, file_format)
