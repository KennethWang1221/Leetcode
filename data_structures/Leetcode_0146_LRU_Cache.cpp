#include <iostream>
#include <unordered_map>
using namespace std;

// Node class for doubly linked list
class Node {
public:
    int key, val;
    Node* prev;
    Node* next;
    Node(int k, int v) : key(k), val(v), prev(nullptr), next(nullptr) {}
};

// LRUCache class
class LRUCache {
private:
    int capacity;
    unordered_map<int, Node*> cache;  // Map from key to node
    Node* left;  // Dummy head
    Node* right;  // Dummy tail

    // Helper function to remove a node from the linked list
    void remove(Node* node) {
        Node* prev = node->prev;
        Node* next = node->next;
        prev->next = next;
        next->prev = prev;
    }

    // Helper function to insert a node at the right end (most recently used)
    void insert(Node* node) {
        Node* prev = right->prev;
        Node* next = right;
        prev->next = node;
        next->prev = node;
        node->prev = prev;
        node->next = next;
    }

public:
    // Constructor to initialize the LRUCache with a given capacity
    LRUCache(int capacity) {
        this->capacity = capacity;
        left = new Node(0, 0);  // Dummy head
        right = new Node(0, 0);  // Dummy tail
        left->next = right;
        right->prev = left;
    }

    // Get the value associated with the key from the cache
    int get(int key) {
        if (cache.find(key) != cache.end()) {
            Node* node = cache[key];
            remove(node);
            insert(node);  // Move the node to the most recent position
            return node->val;
        }
        return -1;  // If the key is not found, return -1
    }

    // Put a key-value pair into the cache
    void put(int key, int value) {
        if (cache.find(key) != cache.end()) {
            remove(cache[key]);
        }

        Node* newNode = new Node(key, value);
        cache[key] = newNode;
        insert(newNode);  // Insert the new node as the most recent node

        // If the cache exceeds the capacity, remove the least recently used (LRU) node
        if (cache.size() > capacity) {
            Node* lru = left->next;  // The LRU node is the node after the left dummy node
            remove(lru);
            cache.erase(lru->key);  // Remove the LRU node from the cache
            delete lru;  // Free the memory of the LRU node
        }
    }
};

// Function to test the LRUCache class
void test_case() {
    LRUCache cache(2);

    cache.put(1, 1);  // cache is {1=1}
    cache.put(2, 2);  // cache is {1=1, 2=2}
    cout << cache.get(1) << endl;  // returns 1
    cache.put(3, 3);  // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
    cout << cache.get(2) << endl;  // returns -1 (not found)
    cache.put(4, 4);  // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
    cout << cache.get(1) << endl;  // returns -1 (not found)
    cout << cache.get(3) << endl;  // returns 3
    cout << cache.get(4) << endl;  // returns 4
}

int main() {
    test_case();  // Run the test case
    return 0;
}


// g++ -std=c++17 Leetcode_0146_LRU_Cache.cpp -o test
