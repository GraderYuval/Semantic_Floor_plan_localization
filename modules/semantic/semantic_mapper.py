from enum import Enum

class ObjectType(Enum):
    WALL = 0
    WINDOW = 1
    DOOR = 2
    UNKNOWN = 3

object_to_color = {
    ObjectType.WALL: 'black',
    ObjectType.WINDOW: 'blue',
    ObjectType.DOOR: 'red',
    ObjectType.UNKNOWN: 'pink',
}
