from dataclasses import dataclass


@dataclass
class Channel:
    S: int
    L: int
    LL: int
    J: int
    channel: int  # isospin projection times 2

    def __post_init__(self):
        self.check()

    @staticmethod
    def isOdd(val):
        return bool(val % 2)

    def check(self):
        tmp = self.S + self.T
        if not (self.isOdd(tmp + self.L) and self.isOdd(tmp + self.LL)):
            raise ValueError(f"Channel {self} is Pauli forbidden.")

        if abs(self.L - self.LL) not in (0, 2):
            raise ValueError(f"Channel {self} doesn't conserve parity.")

        if not(self.L+self.S >= self.J >= abs(self.L - self.S)):
            raise ValueError(f"Channel {self} doesn't obey angular momentum algebra.")

    @property
    def LdotS(self):
        return 0.5 * (self.J**2 - self.L*(self.L+1) - self.S*(self.S+1)) if self.L == self.LL else 0

    @property
    def T(self):
        return 0 if (self.L+self.S) % 2 else 1

    @property
    def g(self):
        return 2*self.S+1

    _SPECTNOTATIONL = {0: "S", 1: "P", 2: "D", 3: "F", 4: "G", 5: "H", 6: "I"}

    @property
    def Lstr(self):
        if self.L in Channel._SPECTNOTATIONL.keys():
            return Channel._SPECTNOTATIONL[self.L]
        else:
            return f"$(L={self.L})$"

    @property
    def spectNotation(self):
        return f"{self.g}{self.Lstr}{self.J}"

    @property
    def spectNotationTeX(self):
        return f"$^{self.g}${self.Lstr}$_{{{self.J}}}$"

    def __str__(self):
        return f"{self.spectNotation} [ S={self.S}; L={self.L}; LL={self.LL}; J={self.J}; T={self.T}; chan={self.channel} ]"


