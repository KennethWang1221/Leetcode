#include <iostream>
#include <stack>
#include <vector>
using namespace std;

class StockSpanner {
public:
    StockSpanner() {}

    int next(int price) {
        int span = 1;
        // While the stack is not empty and the price at the top of the stack is less than or equal to the current price
        while (!stack.empty() && stack.top().first <= price) {
            span += stack.top().second;
            stack.pop();
        }

        // Push the current price and its span onto the stack
        stack.push({price, span});

        return span;
    }

private:
    stack<pair<int, int>> stack;  // Stack to store pairs of (price, span)
};

int main() {
    StockSpanner spanner;
    vector<int> prices = {100, 80, 60, 70, 60, 75, 85};
    vector<int> expected = {1, 1, 1, 2, 1, 4, 6};  // Expected spans for each price

    cout << "Testing StockSpanner:" << endl;
    for (size_t i = 0; i < prices.size(); ++i) {
        int result = spanner.next(prices[i]);
        cout << "Price: " << prices[i] << " -> Span: " << result 
             << " (expected: " << expected[i] << ")" << endl;
    }

    return 0;
}
// g++ -std=c++17 Leetcode_0901_Online_Stock_Span.cpp -o test