#include <iostream>
#include <unordered_map>
#include <vector>
#include <cstdlib>  // for rand()
using namespace std;

class RandomizedSet {
private:
    unordered_map<int, int> dict;  // Stores value to index mapping
    vector<int> list;  // Stores elements for random access
    
public:
    RandomizedSet() {}

    bool insert(int val) {
        if (dict.find(val) != dict.end()) {
            return false;  // Value already exists
        }

        dict[val] = list.size();
        list.push_back(val);
        return true;
    }

    bool remove(int val) {
        if (dict.find(val) == dict.end()) {
            return false;  // Value does not exist
        }

        int idx = dict[val];
        int last_element = list.back();
        
        list[idx] = last_element;  // Move the last element to the place of the element to be removed
        dict[last_element] = idx;  // Update the index of the last element
        
        list.pop_back();  // Remove the last element
        dict.erase(val);  // Remove the element from the dictionary
        
        return true;
    }

    int getRandom() {
        int random_index = rand() % list.size();
        return list[random_index];
    }
};

int main() {
    RandomizedSet obj;
    
    // Test Case 1
    cout << "Insert 1: " << obj.insert(1) << endl;  // Expected: 1
    cout << "Insert 2: " << obj.insert(2) << endl;  // Expected: 1
    cout << "Insert 1: " << obj.insert(1) << endl;  // Expected: 0
    cout << "Get Random: " << obj.getRandom() << endl;  // Expected: Random (1 or 2)
    cout << "Remove 2: " << obj.remove(2) << endl;  // Expected: 1
    cout << "Get Random: " << obj.getRandom() << endl;  // Expected: 1

    // Test Case 2
    cout << "Remove 2 again: " << obj.remove(2) << endl;  // Expected: 0
    return 0;
}

// g++ -std=c++17 Leetcode_0380_Insert_Delete_GetRandom_O1.cpp -o test