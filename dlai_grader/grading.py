class test_case:
    def __init__(self):
        self.failed = False
        self.msg = ""
        self.want = None
        self.got = None

def print_feedback(cases):
    for i, case in enumerate(cases):
        if case.failed:
            print(f"Test case {i+1} FAILED: {case.msg}")
            print(f"  Want: {case.want}")
            print(f"  Got: {case.got}")
        else:
            print(f"Test case {i+1} PASSED")
