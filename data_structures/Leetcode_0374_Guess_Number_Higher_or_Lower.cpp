#include <iostream>
using namespace std;

// Simulate the guess function for testing purposes
int guess(int num) {
    const int picked = 6; // Simulate the picked number
    if (num > picked) {
        return -1; // The guess is too high
    } else if (num < picked) {
        return 1;  // The guess is too low
    } else {
        return 0;  // Correct guess
    }
}

class Solution {
public:
    int guessNumber(int n) {
        int low = 1, high = n;

        while (low <= high) {
            int mid = low + (high - low) / 2;
            int result = guess(mid);

            if (result == 0) {
                return mid;  // Found the picked number
            } else if (result == -1) {
                high = mid - 1;  // Target is smaller, so adjust high
            } else {
                low = mid + 1;  // Target is larger, so adjust low
            }
        }

        return -1;  // This should never be reached if the problem guarantees a valid pick
    }
};

int main() {
    Solution solution;
    int n = 10;  // Range of guesses from 1 to 10

    int result = solution.guessNumber(n);
    cout << "The picked number is: " << result << endl;  // Expected output: 6

    return 0;
}



// g++ -std=c++17 Leetcode_0374_Guess_Number_Higher_or_Lower.cpp -o test