#include <iostream>
#include <unordered_map>
#include <list>
using namespace std;

// Node class for doubly linked list
class Node {
public:
    int key, value, freq;
    Node* prev;
    Node* next;

    Node(int k, int v) : key(k), value(v), freq(1), prev(nullptr), next(nullptr) {}
};

// Doubly Linked List to store nodes with the same frequency
class DoublyLinkedList {
public:
    Node* head;
    Node* tail;
    int size;

    DoublyLinkedList() {
        head = new Node(0, 0);  // Dummy head node
        tail = new Node(0, 0);  // Dummy tail node
        head->next = tail;
        tail->prev = head;
        size = 0;
    }

    // Insert node at the front
    void insert(Node* node) {
        node->next = head->next;
        node->prev = head;
        head->next->prev = node;
        head->next = node;
        size++;
    }

    // Remove node
    void remove(Node* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
        size--;
    }

    // Pop the last node
    Node* pop() {
        if (size > 0) {
            Node* node = tail->prev;
            remove(node);
            return node;
        }
        return nullptr;
    }
};

// LFU Cache class
class LFUCache {
private:
    int capacity, min_freq;
    unordered_map<int, Node*> cache;  // Map from key to node
    unordered_map<int, DoublyLinkedList> freq_map;  // Map from frequency to DLL of nodes

public:
    LFUCache(int capacity) {
        this->capacity = capacity;
        this->min_freq = 0;
    }

    // Get the value associated with the key and update its frequency
    int get(int key) {
        if (cache.find(key) == cache.end()) {
            return -1;
        }

        Node* node = cache[key];
        update(node);
        return node->value;
    }

    // Insert or update key-value pair in the cache
    void put(int key, int value) {
        if (capacity == 0) {
            return;
        }

        if (cache.find(key) != cache.end()) {
            Node* node = cache[key];
            node->value = value;
            update(node);
        } else {
            if (cache.size() >= capacity) {
                DoublyLinkedList& min_freq_list = freq_map[min_freq];
                Node* node_to_remove = min_freq_list.pop();  // Pop the LRU node from the least frequent list
                if (node_to_remove) {
                    cache.erase(node_to_remove->key);
                    delete node_to_remove;
                }
            }

            Node* new_node = new Node(key, value);
            cache[key] = new_node;
            freq_map[1].insert(new_node);
            min_freq = 1;
        }
    }

private:
    // Helper function to update the frequency of a node
    void update(Node* node) {
        int freq = node->freq;
        freq_map[freq].remove(node);

        if (freq == min_freq && freq_map[freq].size == 0) {
            min_freq++;
        }

        node->freq++;
        freq_map[node->freq].insert(node);
    }
};

// Test case
int main() {
    LFUCache lfu_cache(2);
    lfu_cache.put(1, 1);
    lfu_cache.put(2, 2);
    cout << lfu_cache.get(1) << endl;  // Expected output: 1
    lfu_cache.put(3, 3);  // Evicts key 2
    cout << lfu_cache.get(2) << endl;  // Expected output: -1 (not found)
    cout << lfu_cache.get(3) << endl;  // Expected output: 3
    lfu_cache.put(4, 4);  // Evicts key 1
    cout << lfu_cache.get(1) << endl;  // Expected output: -1 (not found)
    cout << lfu_cache.get(3) << endl;  // Expected output: 3
    cout << lfu_cache.get(4) << endl;  // Expected output: 4

    return 0;
}


// g++ -std=c++17 Leetcode_0460_LFU_Cache.cpp -o test
