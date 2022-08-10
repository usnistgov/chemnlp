from chemnlp.utils.process_doc import ProcessDoc


def test_formula():
    txt = "MgB2 is a superconductor."
    pdoc = ProcessDoc(text=txt)
    print(pdoc.get_chemical_formulae())
