#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <time.h>
#include "third_party/cJSON.h"

void saveToFile(const char *filename, const char *text) {
    FILE *file = fopen(filename, "a"); // Open the file for appending
    if (file == NULL) {
        perror("Unable to open the file for writing");
        exit(1);
    }
    fprintf(file, "%s\n", text); // Append the L-system string to the file, followed by a newline
    fclose(file); // Close the file
}

// Define the rules of the L-system with probabilities
typedef struct {
    char *predecessor;
    char *successor;
    double probability; // Probability of the rule being chosen
} Rule;

int startsWith(const char *pre, const char *str) {
    size_t lenpre = strlen(pre), lenstr = strlen(str);
    return lenstr < lenpre ? 0 : strncmp(pre, str, lenpre) == 0;
}

// Function to apply stochastic rules to generate the next iteration of the system
// char *applyStochasticRules(char *s, Rule *rules, int numRules) {

//     // Create a buffer for the new string
//     char *buffer = malloc(strlen(s) * 10 + 1); // Assume each character can be at most 10 characters long after rule application
//     buffer[0] = '\0'; // Start with an empty string

//     // Go through each character of the input string and apply rules
//     for (int i = 0; s[i] != '\0'; i++) {
//         double roll = (double)rand() / (double)RAND_MAX;
//         double cumulativeProbability = 0.0;

//         // Check if the current character matches any rule predecessor and apply based on probability
//         for (int j = 0; j < numRules; j++) {
//             if (s[i] == rules[j].predecessor[0]) { // Assuming single character predecessor
//                 cumulativeProbability += rules[j].probability;
//                 if (roll <= cumulativeProbability) {
//                     // Append the successor string to the buffer
//                     strcat(buffer, rules[j].successor);
//                     break;
//                 }
//             }
//         }
//         // If no rule was applied, just copy the character
//         if (roll > cumulativeProbability) {
//             int len = strlen(buffer);
//             buffer[len] = s[i];
//             buffer[len + 1] = '\0';
//         }
//     }
//     return buffer;
// }

char *applyStochasticRules(char *s, Rule *rules, int numRules) {
    char *buffer = malloc(strlen(s) * 10 + 1);
    buffer[0] = '\0';

    for (int i = 0; s[i] != '\0';) {
        double roll = (double)rand() / (double)RAND_MAX;
        double cumulativeProbability = 0.0;
        int ruleApplied = 0;

        for (int j = 0; j < numRules; j++) {
            if (startsWith(rules[j].predecessor, &s[i])) {
                cumulativeProbability += rules[j].probability;
                if (roll <= cumulativeProbability) {
                    strcat(buffer, rules[j].successor);
                    i += strlen(rules[j].predecessor); // Advance the index by the length of the predecessor
                    ruleApplied = 1;
                    break;
                }
            }
        }

        if (!ruleApplied) {
            int len = strlen(buffer);
            buffer[len] = s[i];
            buffer[len + 1] = '\0';
            i++; // Advance the index by 1
        }
    }
    return buffer;
}


// Function to load L-system from a JSON file
void loadLSystem(const char *filename, char **axiom, Rule **rules, int *numRules) {
    // Open the JSON file
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Unable to open the file");
        exit(1);
    }

    // Read the file into a string
    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);
    char *data = malloc(length + 1);
    fread(data, 1, length, file);
    data[length] = '\0';
    fclose(file);

    // Parse the JSON data
    cJSON *json = cJSON_Parse(data);
    if (json == NULL) {
        const char *error_ptr = cJSON_GetErrorPtr();
        if (error_ptr != NULL) {
            fprintf(stderr, "Error before: %s\n", error_ptr);
        }
        exit(1);
    }

    // Extract the axiom
    cJSON *jsonAxiom = cJSON_GetObjectItemCaseSensitive(json, "axiom");
    *axiom = strdup(jsonAxiom->valuestring);

    // Extract the rules
    cJSON *jsonRules = cJSON_GetObjectItemCaseSensitive(json, "rules");
    *numRules = cJSON_GetArraySize(jsonRules);
    *rules = malloc((*numRules) * sizeof(Rule));

    for (int i = 0; i < *numRules; i++) {
        cJSON *jsonRule = cJSON_GetArrayItem(jsonRules, i);
        cJSON *predecessor = cJSON_GetObjectItemCaseSensitive(jsonRule, "predecessor");
        cJSON *successor = cJSON_GetObjectItemCaseSensitive(jsonRule, "successor");
        cJSON *probability = cJSON_GetObjectItemCaseSensitive(jsonRule, "probability");

        (*rules)[i].predecessor = strdup(predecessor->valuestring);
        (*rules)[i].successor = strdup(successor->valuestring);
        (*rules)[i].probability = probability->valuedouble;
    }

    // Clean up
    cJSON_Delete(json);
    free(data);
}

int main(int argc, char** argv) {
    
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <JSON file name> <number of samples>\n", argv[0]);
        return 1;
    }
    // Seed the random number generator
    srand((unsigned int)time(NULL));

    // Construct the path to the JSON file within the axioms_and_rules directory
    char jsonFilePath[512];
    snprintf(jsonFilePath, sizeof(jsonFilePath), "axioms_and_rules/%s.json", argv[1]);

    int iterations = atoi(argv[2]);
    int n = atoi(argv[3]); // Number of samples to generate

    // Extract the filename without extension for use in output filenames
    char *jsonFilename = argv[1];
    char baseFilename[256];
    char *dot = strrchr(jsonFilename, '.');
    if (dot) {
        snprintf(baseFilename, dot - jsonFilename + 1, "%s", jsonFilename); // Copy the filename without extension
    } else {
        strcpy(baseFilename, jsonFilename); // Use the whole filename as it has no extension
    }

    // Construct the path to the output file within the data directory
    char outputFilename[512];
    snprintf(outputFilename, sizeof(outputFilename), "data/%s_samples.txt", baseFilename);

    // Load the axiom and rules from the JSON file
    char *axiom;
    Rule *rules;
    int numRules;
    loadLSystem(jsonFilePath, &axiom, &rules, &numRules);

    // Open the output file once and pass the FILE* to saveToFile
    FILE *file = fopen(outputFilename, "w"); // Open the file for writing (use "w" to overwrite existing files)
    if (file == NULL) {
        perror("Unable to open the output file for writing");
        exit(1);
    }

    // Generate L-system string for a few iterations
    for (int sample = 0; sample < n; ++sample) {
        char *current = strdup(axiom);
        for (int i = 0; i < iterations; i++) {
            char *next = applyStochasticRules(current, rules, numRules);
            free(current);
            current = next;
        }

        // Append the generated string to the file
        fprintf(file, "%s\n", current); // Append the L-system string to the file, followed by a newline

        free(current); // Free the string after saving to file
    }

    // Close the file after all samples have been written
    fclose(file);

    // Clean up
    free(axiom);
    for (int i = 0; i < numRules; i++) {
        free(rules[i].predecessor);
        free(rules[i].successor);
    }
    free(rules);

    return 0;
}
