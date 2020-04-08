import dis


def calculate_complexity(environment, randomness=0):
    bytecode = dis.Bytecode(environment.calculate_percept).dis()
    return len([line for line in bytecode.splitlines() if line]) * 2
