# -*- coding: utf-8 -*-
import numpy as np
import ctypes
from OpenGL.GL import *

from .renderbuffer import Renderbuffer, RenderbufferMultisample
from .texture import Texture, TextureMultisample


class Framebuffer(object):
    def __init__(self, attachements):
        self.__id = np.empty(1, dtype=np.uint32)
        glCreateFramebuffers(len(self.__id), self.__id)
        for k in list(attachements.keys()):
            attachement = attachements[k]
            if isinstance(attachement, Renderbuffer) or isinstance(attachement, RenderbufferMultisample):
                glNamedFramebufferRenderbuffer(ctypes.c_uint(self.id), k, GL_RENDERBUFFER, attachement.id)
            elif isinstance(attachement, Texture) or isinstance(attachement, TextureMultisample):
                glNamedFramebufferTexture(ctypes.c_uint(self.id), k, attachement.id, 0)
            else:
                raise ValueError("Unknown frambuffer attachement class: {0}".format(attachement))

        if glCheckNamedFramebufferStatus(ctypes.c_uint(self.id), GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Framebuffer not complete.")
        self.__attachements = attachements

    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.id)

    def delete(self):
        glDeleteFramebuffers(1, self.id)
        for k in list(self.__attachements.keys()):
            self.__attachements[k].delete()

    @property
    def id(self):
        return self.__id[0]
