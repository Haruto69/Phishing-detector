#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Define the structure as TrieNode
struct TrieNode {
    struct TrieNode* children[26]; // Array for 26 lowercase letters
    int count;                     // To store the frequency of the word
};

// Define NODE as a pointer to struct TrieNode
typedef struct TrieNode* NODE;

// Exported function to create a new NODE
__declspec(dllexport) NODE createNode() {
    NODE node = (NODE)malloc(sizeof(struct TrieNode));
    node->count = 0;
    for (int i = 0; i < 26; i++) {
        node->children[i] = NULL;
    }
    return node;
}

// Exported function to insert a word into the Trie
__declspec(dllexport) void insert(NODE root, const char* word) {
    NODE current = root;
    for (int i = 0; word[i] != '\0'; i++) {
        int index = word[i] - 'a'; // Map 'a' to 0, 'b' to 1, ..., 'z' to 25
        if (current->children[index] == NULL) {
            current->children[index] = createNode();
        }
        current = current->children[index];
    }
    current->count += 1; // Increment the word frequency
}

// Exported function to get the feature vector for a word
__declspec(dllexport) void getFeatureVector(NODE root, const char* word, int* feature_vector, int index) {
    NODE current = root;
    for (int i = 0; word[i] != '\0'; i++) {
        int charIndex = word[i] - 'a';
        if (current->children[charIndex] == NULL) {
            feature_vector[index] = 0;
            return;
        }
        current = current->children[charIndex];
    }
    feature_vector[index] = current->count; // Store the word frequency
}

// Exported function to cleanup the Trie (for DLL memory management)
__declspec(dllexport) void freeTrie(NODE root) {
    if (root == NULL) return;
    for (int i = 0; i < 26; i++) {
        if (root->children[i] != NULL) {
            freeTrie(root->children[i]);
        }
    }
    free(root);
}
