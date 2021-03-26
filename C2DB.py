import ase.db as db
from ase.io.vasp import write_vasp
from os import makedirs, remove
from os.path import exists
from shutil import rmtree
import subprocess
from subprocess import PIPE


ResultDirPath = "C2DBResult/"
StiffnessNameArray = ["c_11", "c_12", "c_13", "c_22", "c_23", "c_33"]


Database = db.connect("Data/c2db.db")
for Row in Database.select("magstate=NM"):
    if Row.get("has_asr_stiffness") == None:
        pass
    else:
        Formula = Row.formula
        CurrentResultDirPath = ResultDirPath + Formula + "/"

        if exists(CurrentResultDirPath):
            rmtree(CurrentResultDirPath)
        makedirs(CurrentResultDirPath)

        with open(CurrentResultDirPath + "Stiffness.dat", "w") as IO:
            for (index, PropertyName) in enumerate(StiffnessNameArray):
                IO.write(str(Row.get(PropertyName)))

                if index < 5:
                    IO.write(",")

        TargetPOSCARPath = CurrentResultDirPath + "POSCAR"
        write_vasp(
            TargetPOSCARPath,
            Row.toatoms(),
            direct = True,
            vasp5 = True
        )

        LatticeCommand = "sed -n '3,5p' " + TargetPOSCARPath
        PositionCommand = "sed -n '6,$p' " + TargetPOSCARPath
        MaterialLattice = subprocess.Popen(
            LatticeCommand,
            shell = True,
            stdout = PIPE
        ).stdout.read().decode()
        MaterialPosition = subprocess.Popen(
            PositionCommand,
            shell = True,
            stdout = PIPE
        ).stdout.read().decode()
        with open(CurrentResultDirPath + "Lattice.dat", "w") as resultIO:
            resultIO.write(MaterialLattice)
        with open(CurrentResultDirPath + "Position.dat", "w") as resultIO:
            resultIO.write(MaterialPosition)
        remove(TargetPOSCARPath)

        print(Formula)