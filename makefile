CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

example: ExampleCode.cpp
	g++ ExampleCode.cpp -o Example.exec -g $(CFLAGS) $(LIBS) 
