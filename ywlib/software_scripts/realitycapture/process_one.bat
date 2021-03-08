rem set PATH=%PATH%;C:\Program Files\Capturing Reality\RealityCapture\
"C:\Program Files\Capturing Reality\RealityCapture\RealityCapture.exe" -addFolder %1 -align  -setReconstructionRegion %2  -save %3 -quit 
timeout 5 > NUL
"C:\Program Files\Capturing Reality\RealityCapture\RealityCapture.exe" -load %3 -selectMaximalComponent -mvs -simplify 1000000 -smooth -calculateVertexColors -calculateTexture -exportModel "Model 3" %4 F:\DATA\rc_batch\objParams.xml -save %5 -quit 
timeout 5 > NUL 