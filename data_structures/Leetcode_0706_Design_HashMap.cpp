#include <iostream>
#include <vector>  // Include the vector header

class Node {
public:
    int key;
    int val;
    Node* next;

    Node(int k = -1, int v = -1, Node* n = nullptr) : key(k), val(v), next(n) {}
};

class MyHashMap {
private:
    std::vector<Node*> map;  // Add std:: here to fully qualify vector

public:
    MyHashMap() {
        map.resize(1000);
        for (int i = 0; i < 1000; ++i) {
            map[i] = new Node();
        }
    }

    int hash(int key) {
        return key % 1000;
    }

    void put(int key, int value) {
        int hash_key = hash(key);
        Node* cur = map[hash_key];

        while (cur->next) {
            if (cur->next->key == key) {
                cur->next->val = value;
                return;
            }
            cur = cur->next;
        }

        cur->next = new Node(key, value);
    }

    int get(int key) {
        int hash_key = hash(key);
        Node* cur = map[hash_key];

        while (cur->next) {
            if (cur->next->key == key) {
                return cur->next->val;
            }
            cur = cur->next;
        }

        return -1;
    }

    void remove(int key) {
        int hash_key = hash(key);
        Node* cur = map[hash_key];

        while (cur->next) {
            if (cur->next->key == key) {
                Node* temp = cur->next;
                cur->next = cur->next->next;
                delete temp;
                return;
            }
            cur = cur->next;
        }
    }
};

/**
 * Your MyHashMap object will be instantiated and called as such:
 * MyHashMap* obj = new MyHashMap();
 * obj->put(key,value);
 * int param_2 = obj->get(key);
 * obj->remove(key);
 */

int main() {
    MyHashMap* hashmap = new MyHashMap();

    // Test put and get methods
    hashmap->put(1, 100);
    hashmap->put(2, 200);
    std::cout << "Get key 1: " << hashmap->get(1) << std::endl;  // Expected output: 100
    std::cout << "Get key 2: " << hashmap->get(2) << std::endl;  // Expected output: 200
    std::cout << "Get key 3: " << hashmap->get(3) << std::endl;  // Expected output: -1 (not found)

    // Test put method with the same key, it should update the value
    hashmap->put(1, 500);
    std::cout << "Get key 1 after update: " << hashmap->get(1) << std::endl;  // Expected output: 500

    // Test remove method
    hashmap->remove(2);
    std::cout << "Get key 2 after removal: " << hashmap->get(2) << std::endl;  // Expected output: -1 (not found)

    // Test remove method with a key that does not exist
    hashmap->remove(3);
    std::cout << "Get key 3 after removal: " << hashmap->get(3) << std::endl;  // Expected output: -1 (not found)

    delete hashmap;  // Clean up memory
    return 0;
}

// g++ -std=c++17 Leetcode_0706_Design_HashMap.cpp -o test