#include <iostream>
using namespace std;

class Node {
public:
    int val;
    Node* next;
    Node(int x) : val(x), next(nullptr) {}
};

class MyCircularQueue {
private:
    Node* head;
    Node* tail;
    int capacity;
    int size;

public:
    MyCircularQueue(int k) {
        head = tail = nullptr;
        capacity = k;
        size = 0;
    }

    bool enQueue(int value) {
        if (isFull()) {
            return false;
        }

        Node* node = new Node(value);
        if (size == 0) {
            head = tail = node;
        } else {
            tail->next = node;
            tail = node;
        }

        size++;
        return true;
    }

    bool deQueue() {
        if (isEmpty()) {
            return false;
        }

        head = head->next;
        size--;
        return true;
    }

    int Front() {
        return isEmpty() ? -1 : head->val;
    }

    int Rear() {
        return isEmpty() ? -1 : tail->val;
    }

    bool isEmpty() {
        return size == 0;
    }

    bool isFull() {
        return size == capacity;
    }
};

// Function to test MyCircularQueue with different operations
void test_case() {
    MyCircularQueue queue(3);

    cout << queue.enQueue(1) << endl; // Expected: 1 (True)
    cout << queue.enQueue(2) << endl; // Expected: 1 (True)
    cout << queue.enQueue(3) << endl; // Expected: 1 (True)
    cout << queue.enQueue(4) << endl; // Expected: 0 (False, queue is full)

    cout << queue.Front() << endl; // Expected: 1
    cout << queue.Rear() << endl;  // Expected: 3

    cout << queue.deQueue() << endl; // Expected: 1 (True)

    cout << queue.Front() << endl; // Expected: 2
    cout << queue.Rear() << endl;  // Expected: 3

    cout << queue.enQueue(4) << endl; // Expected: 1 (True)

    cout << queue.Front() << endl; // Expected: 2
    cout << queue.Rear() << endl;  // Expected: 4

    cout << queue.deQueue() << endl; // Expected: 1 (True)
    cout << queue.deQueue() << endl; // Expected: 1 (True)
    cout << queue.deQueue() << endl; // Expected: 1 (True)
    cout << queue.deQueue() << endl; // Expected: 0 (False, queue is empty)
}

int main() {
    test_case();
    return 0;
}


// g++ -std=c++17 Leetcode_0622_Design_Circular_Queue.cpp -o test
