from julia.api import Julia
jl = Julia(compiled_modules=False)
print(jl.eval('1 + 1'))
