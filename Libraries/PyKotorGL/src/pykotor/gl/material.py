"""TXI material bindings shared by mesh draws."""
from __future__ import annotations

from typing import TYPE_CHECKING

from glm import vec3
from OpenGL.GL import (GL_BLEND, GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_POLYGON_OFFSET_FILL,
                       GL_SRC_ALPHA, GL_TEXTURE0, GL_TEXTURE_2D, GL_TEXTURE_3D,
                       GL_TEXTURE_CUBE_MAP, glActiveTexture, glBlendFunc, glDepthMask,
                       glDisable, glEnable, glPolygonOffset)

if TYPE_CHECKING:
    from pykotor.gl.scene import Scene
    from pykotor.gl.models.mdl import Mesh


class Material:
    def __init__(self, scene: Scene, diffuse: str, lightmap: str):
        self.scene = scene
        self.diffuse_name, self.lightmap_name = diffuse, lightmap

    def refresh(self):
        self.diffuse = self.scene.loadTexture(self.diffuse_name)
        self.lightmap = self.scene.loadTexture(self.lightmap_name, lightmap=True)
        self.fields = self.diffuse.settings.material
        self.self_normal = self.diffuse.settings.texture["isbumpmap"] == 2
        bump_name = self.fields["bumpmaptexture"]
        self.has_bump = self.self_normal or bool(bump_name and bump_name.upper() != "NULL")
        self.bump = (self.diffuse if self.self_normal else self.scene.loadTexture(bump_name, normal=True)
                     if self.has_bump else self.scene.blank_normal)
        env_name = self.fields["envmaptexture"] or self.fields["bumpyshinytexture"]
        self.has_environment = bool(env_name and env_name.upper() != "NULL")
        self.environment = self.scene.loadTexture(env_name) if self.has_environment else self.scene.blank_texture
        self.transparent = (self.fields["blending"] == "additive" or self.fields["wateralpha"] < 1
                            or self.diffuse.settings.texture["useglobalalpha"] and self.scene.global_alpha < 1
                            or not self.has_environment and self.diffuse.transparent and self.fields["blending"] != "punchthrough")

    @staticmethod
    def _bind(unit, texture):
        glActiveTexture(GL_TEXTURE0 + unit)
        texture.use()

    def bind(self, mesh: Mesh, *, color: bool):
        scene, shader = self.scene, self.scene.shader
        if color:
            for texture in (self.diffuse, self.lightmap, self.bump, self.environment):
                texture.advance(scene.frame_index, scene.frame_delta)
        diffuse_kind = {GL_TEXTURE_2D: 0, GL_TEXTURE_CUBE_MAP: 1, GL_TEXTURE_3D: 2}[self.diffuse.target]
        self._bind((0, 5, 7)[diffuse_kind], self.diffuse)
        self._bind(1, self.lightmap)
        self._bind(2, self.bump)
        self._bind(3 if self.environment.target == GL_TEXTURE_CUBE_MAP else 4, self.environment)
        shader.set_int("diffuseKind", diffuse_kind)
        shader.set_bool("hasNormals", mesh.has_normals)
        shader.set_bool("enableLightmap", scene.use_lightmap and self.lightmap_name.upper() != "NULL")
        shader.set_bool("hasBump", self.has_bump)
        shader.set_bool("hasEnvironment", self.has_environment)
        shader.set_bool("environmentIsCube", self.environment.target == GL_TEXTURE_CUBE_MAP)
        shader.set_bool("selfNormal", self.self_normal)
        shader.set_bool("objectSpaceNormal", self.bump.settings.texture["isbumpmap"] == 2)
        shader.set_bool("diffuseBump", self.bump.settings.texture["isdiffusebumpmap"])
        shader.set_bool("specularBump", self.bump.settings.texture["isspecularbumpmap"])
        shader.set_bool("punchthrough", self.fields["blending"] == "punchthrough")
        shader.set_bool("useGlobalAlpha", self.diffuse.settings.texture["useglobalalpha"])
        shader.set_float("globalAlpha", scene.global_alpha)
        shader.set_float("waterAlpha", self.fields["wateralpha"])
        shader.set_int("renderBMLMType", self.fields["renderbmlmtype"])
        shader.set_float("envmapAlpha", self.diffuse.settings.texture["envmapalpha"])
        shader.set_float("bumpIntensity", self.bump.settings.texture["bumpintensity"])
        shader.set_float("diffuseIntensity", self.bump.settings.texture["diffusebumpintensity"])
        shader.set_float("specularIntensity", self.bump.settings.texture["specularbumpintensity"])
        shader.set_vector3("specularColor", vec3(*self.bump.settings.texture["specularcolor"]))
        shader.set_int("dirtFormula", -1)
        if mesh.dirt_index:
            dirt = scene.dirt_texture(mesh.dirt_index)
            self._bind(6, dirt)
            shader.set_int("dirtFormula", scene.dirty_parameters[mesh.dirt_index - 1][2])
            shader.set_bool("dirtWorldspace", mesh.dirt_worldspace)
        if color and self.transparent:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE if self.fields["blending"] == "additive" else GL_ONE_MINUS_SRC_ALPHA)
            glDepthMask(False)
        else:
            glDisable(GL_BLEND)
            glDepthMask(True)
        if self.fields["decal"]:
            glEnable(GL_POLYGON_OFFSET_FILL)
            glPolygonOffset(-1, -1)

    @staticmethod
    def finish():
        glDepthMask(True)
        glDisable(GL_BLEND)
        glDisable(GL_POLYGON_OFFSET_FILL)
