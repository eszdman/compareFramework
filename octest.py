from oct2py import Oct2Py

oc = Oct2Py()

script = "function y = myScript(x)\n" \
         "    y = x-5" \
         "end"

with open("myScript.m", "w+") as f:
    f.write(script)

oc.myScript(7)
