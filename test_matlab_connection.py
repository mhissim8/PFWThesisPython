import matlab.engine

print("Starting MATLAB R2025b...")

eng = matlab.engine.start_matlab()

try:
    print("MATLAB version:", eng.version())
    print("MATLAB calculation:", eng.sqrt(25.0))
    print("MATLAB-Python connection successful!")
finally:
    eng.quit()
