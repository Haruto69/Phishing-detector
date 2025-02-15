#ifndef MAIN_H
#define MAIN_H

// Define the structure as TrieNode
struct TrieNode {
    struct TrieNode* children[26]; // Array for 26 lowercase letters
    int count;                     // To store the frequency of the word
};

// Define NODE as a pointer to struct TrieNode
typedef struct TrieNode* NODE;

// Exported function to create a new NODE
__declspec(dllexport) NODE createNode();

// Exported function to insert a word into the Trie
__declspec(dllexport) void insert(NODE root, const char* word);

// Exported function to get the feature vector for a word
__declspec(dllexport) void getFeatureVector(NODE root, const char* word, int* feature_vector, int index);

// Exported function to cleanup the Trie (for DLL memory management)
__declspec(dllexport) void freeTrie(NODE root);

#endif // MAIN_H
