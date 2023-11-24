all:
	gcc ./LSystemGenerator.c ./third_party/cJSON.c -o LSystemGenerator

clean:
	rm LSystemGenerator.exe