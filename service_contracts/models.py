from typing import Literal

from pydantic import BaseModel


class ModelDescription(BaseModel):
    id: str
    object: Literal['model'] = 'model'
    owned_by: Literal['local'] = 'local'


class ModelList(BaseModel):
    object: Literal['list'] = 'list'
    data: list[ModelDescription]
