from dataclasses import dataclass

@dataclass(frozen=False, eq=True)
class AddedToken:
    def __init__(
        self, content: str, special=False
    ):
        self.content = content
        self.special = special

    def __getstate__(self):
        return self.__dict__

    def __str__(self):
        return self.content