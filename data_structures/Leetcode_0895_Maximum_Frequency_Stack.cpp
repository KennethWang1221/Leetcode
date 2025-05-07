#include <iostream>
#include <unordered_map>
#include <stack>
#include <vector>
using namespace std;

class FreqStack {
public:
    FreqStack() {}

    void push(int val) {
        int valCnt = ++cnt[val];
        if (valCnt > maxCnt) {
            maxCnt = valCnt;
        }
        stacks[valCnt].push(val);
    }

    int pop() {
        int res = stacks[maxCnt].top();
        stacks[maxCnt].pop();
        cnt[res]--;
        
        if (stacks[maxCnt].empty()) {
            maxCnt--;
        }

        return res;
    }

private:
    unordered_map<int, int> cnt;          // Stores the frequency of each element
    unordered_map<int, stack<int>> stacks; // Maps the frequency to a stack of elements with that frequency
    int maxCnt = 0;  // Tracks the maximum frequency
};

int main() {
    FreqStack freqStack;

    freqStack.push(5);
    freqStack.push(7);
    freqStack.push(5);
    freqStack.push(7);
    freqStack.push(4);
    freqStack.push(5);

    cout << freqStack.pop() << endl; // Expected output: 5
    cout << freqStack.pop() << endl; // Expected output: 7
    cout << freqStack.pop() << endl; // Expected output: 5

    return 0;
}
// g++ -std=c++17 Leetcode_0895_Maximum_Frequency_Stack.cpp -o test