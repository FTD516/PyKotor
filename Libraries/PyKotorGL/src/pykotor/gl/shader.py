from __future__ import annotations

from typing import TYPE_CHECKING

import glm
from OpenGL.GL import (GL_FALSE, GL_FRAGMENT_SHADER, GL_VERTEX_SHADER, glGetUniformLocation,
                       glUniform1f, glUniform1i, glUniform3fv, glUniform4fv, glUniformMatrix4fv,
                       glUseProgram, shaders)

from pykotor.gl.texture import Texture

if TYPE_CHECKING:
    from glm import mat4, vec3, vec4

KOTOR_VSHADER = """
#version 330 core

layout (location = 0) in vec3 flags;
layout (location = 1) in vec3 position;
layout (location = 2) in vec3 normal;
layout (location = 3) in vec3 uv;
layout (location = 4) in vec3 uv2;
layout (location = 5) in vec3 localPosition;

out vec2 diffuse_uv;
out vec2 lightmap_uv;
out vec3 world_position;
out vec3 world_normal;
out vec3 local_position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    vec4 world = model * vec4(position, 1.0);
    gl_Position = projection * view * world;
    diffuse_uv = vec2(uv.x, uv.y);
    lightmap_uv = vec2(uv2.x, uv2.y);
    world_position = world.xyz;
    world_normal = transpose(inverse(mat3(model))) * normal;
    local_position = localPosition;
}
"""


KOTOR_FSHADER = """
#version 420
in vec2 diffuse_uv;
in vec2 lightmap_uv;
in vec3 world_position;
in vec3 world_normal;
in vec3 local_position;

out vec4 FragColor;

layout(binding = 0) uniform sampler2D diffuse;
layout(binding = 1) uniform sampler2D lightmap;
layout(binding = 2) uniform sampler2D bumpmap;
layout(binding = 3) uniform samplerCube environment_cube;
layout(binding = 4) uniform sampler2D environment_2d;
layout(binding = 5) uniform samplerCube diffuse_cube;
layout(binding = 6) uniform sampler3D dirt_volume;
layout(binding = 7) uniform sampler3D diffuse_volume;
uniform bool enableLightmap;
uniform bool hasNormals;
uniform bool hasBump;
uniform bool hasEnvironment;
uniform bool environmentIsCube;
uniform bool diffuseBump;
uniform bool specularBump;
uniform bool objectSpaceNormal;
uniform bool selfNormal;
uniform bool punchthrough;
uniform bool useGlobalAlpha;
uniform bool dirtWorldspace;
uniform int diffuseKind;
uniform int dirtFormula;
uniform int renderBMLMType;
uniform float waterAlpha;
uniform float globalAlpha;
uniform float envmapAlpha;
uniform float bumpIntensity;
uniform float diffuseIntensity;
uniform float specularIntensity;
uniform vec3 specularColor;
uniform vec3 eyePosition;
uniform vec3 lightDirection;
uniform vec3 lightColor;
uniform vec3 ambientColor;
uniform mat4 model;

vec3 reflected_normal(vec3 normal, vec3 viewDirection)
{
    return reflect(-viewDirection, normal);
}

vec3 environment(vec3 reflected)
{
    if (environmentIsCube) return texture(environment_cube, reflected).rgb;
    vec3 shifted = reflected + vec3(0.0, 0.0, 1.0);
    vec2 sphere = reflected.xy / (2.0 * max(length(shifted), 1e-12)) + 0.5;
    return texture(environment_2d, sphere).rgb;
}

mat3 tangent_frame(vec3 normal)
{
    vec3 px = dFdx(world_position), py = dFdy(world_position);
    vec2 tx = dFdx(diffuse_uv), ty = dFdy(diffuse_uv);
    vec3 pyPerp = cross(py, normal), pxPerp = cross(normal, px);
    vec3 tangent = pyPerp * tx.x + pxPerp * ty.x;
    vec3 bitangent = pyPerp * tx.y + pxPerp * ty.y;
    float inverse = inversesqrt(max(max(dot(tangent, tangent), dot(bitangent, bitangent)), 1e-20));
    return mat3(tangent * inverse, bitangent * inverse, normal);
}

void main()
{
    vec3 normal = normalize(hasNormals ? world_normal : cross(dFdx(world_position), dFdy(world_position)));
    vec3 viewDirection = normalize(eyePosition - world_position);
    vec4 base;
    if (diffuseKind == 1) base = texture(diffuse_cube, reflected_normal(normal, viewDirection));
    else if (diffuseKind == 2) base = texture(diffuse_volume, world_position);
    else base = texture(diffuse, diffuse_uv);
    float alpha = selfNormal ? 1.0 : base.a;
    if (punchthrough && alpha <= 0.35) discard;
    if (alpha <= 0.0 && !hasEnvironment && !selfNormal) discard;
    vec3 color = selfNormal ? vec3(1.0) : base.rgb;
    vec3 bumped = normal;
    float normalWeight = 1.0;
    if (hasBump) {
        vec4 sampleNormal = texture(bumpmap, diffuse_uv);
        normalWeight = sampleNormal.a;
        if (objectSpaceNormal) {
            bumped = normalize(transpose(inverse(mat3(model))) * (sampleNormal.rgb * 2.0 - 1.0));
        } else {
            vec3 tangentNormal = vec3(-(sampleNormal.g * 255.0 - 128.0),
                                      sampleNormal.b * 255.0 - 128.0,
                                      sampleNormal.r * 255.0 - 128.0) / 127.0;
            tangentNormal.xy *= bumpIntensity;
            bumped = normalize(tangent_frame(normal) * tangentNormal);
        }
        if (diffuseBump && (!enableLightmap || renderBMLMType == 0)) {
            color *= ambientColor + lightColor * max(dot(bumped, lightDirection), 0.0) * diffuseIntensity;
        }
    }
    if (hasEnvironment) {
        vec3 reflection = environment(reflected_normal(bumped, viewDirection)) * envmapAlpha;
        color = enableLightmap ? color + reflection * (1.0 - alpha) : mix(reflection, color, alpha);
        alpha = 1.0;
    }
    if (dirtFormula >= 0) {
        vec3 coordinates;
        if (dirtWorldspace) {
            coordinates = world_position * 1.5;
            coordinates = coordinates * 0.25 + coordinates.yzx;
            coordinates = coordinates * 0.125 + coordinates.zxy;
            coordinates *= 1.0675;
        } else coordinates = local_position + vec3(diffuse_uv, 0.0);
        vec3 dirt = texture(dirt_volume, coordinates).rgb;
        if (dirtFormula == 0) color -= dirt;
        else if (dirtFormula == 1) color += dirt;
        else color *= dirt;
    }
    if (enableLightmap) color *= texture(lightmap, lightmap_uv).rgb;
    if (hasBump && specularBump && (!enableLightmap || renderBMLMType != 0)) {
        vec3 halfway = normalize(lightDirection + viewDirection);
        float specular = pow(max(dot(bumped, halfway), 0.0), 32.0);
        color += lightColor * specularColor * specular * specularIntensity * normalWeight;
    }
    alpha *= waterAlpha;
    if (useGlobalAlpha) alpha *= globalAlpha;
    FragColor = vec4(color, alpha);
}
"""


PICKER_VSHADER = """
#version 330 core

layout (location = 1) in vec3 position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model *  vec4(position, 1.0);
}
"""


PICKER_FSHADER = """
#version 330

uniform vec3 colorId;

out vec4 FragColor;

void main()
{
    FragColor = vec4(colorId, 1.0);
}
"""


PLAIN_VSHADER = """
#version 330 core

layout (location = 1) in vec3 position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model *  vec4(position, 1.0);
}
"""


PLAIN_FSHADER = """
#version 330

uniform vec4 color;

out vec4 FragColor;

void main()
{
    FragColor = color;
}
"""


class Shader:
    def __init__(self, vshader: str, fshader: str):
        vertex_shader = shaders.compileShader(vshader, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(fshader, GL_FRAGMENT_SHADER)
        self._id: int = shaders.compileProgram(vertex_shader, fragment_shader)

    def use(self):
        glUseProgram(self._id)

    def uniform(self, uniform_name: str):
        return glGetUniformLocation(self._id, uniform_name)

    def set_matrix4(self, uniform: str, matrix: mat4):
        glUniformMatrix4fv(self.uniform(uniform), 1, GL_FALSE, glm.value_ptr(matrix))

    def set_vector4(self, uniform: str, vector: vec4):
        glUniform4fv(self.uniform(uniform), 1, glm.value_ptr(vector))

    def set_vector3(self, uniform: str, vector: vec3):
        glUniform3fv(self.uniform(uniform), 1, glm.value_ptr(vector))

    def set_bool(self, uniform: str, boolean: bool):
        glUniform1i(self.uniform(uniform), boolean)

    def set_int(self, uniform: str, value: int):
        glUniform1i(self.uniform(uniform), value)

    def set_float(self, uniform: str, value: float):
        glUniform1f(self.uniform(uniform), value)
