import subprocess

# generate finite state machine pdf
with open('fsm.dot', 'r') as f:
    dot_code = f.read()
    
subprocess.run(['dot', '-Tpdf', 'fsm.dot', '-o', 'fsm.pdf'])
print("PDF saved as fsm.pdf")