#include <iostream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

class Codec {
public:
    // Encodes a list of strings to a single string.
    string encode(const vector<string>& strs) {
        string res = "";
        for (const string& s : strs) {
            res += to_string(s.length()) + "#" + s;  // Encode length and string
        }
        return res;
    }

    // Decodes a single string to a list of strings.
    vector<string> decode(const string& s) {
        vector<string> res;
        int i = 0;

        while (i < s.length()) {
            int j = i;
            // Find the position of the '#' separator
            while (s[j] != '#') {
                j++;
            }
            int length = stoi(s.substr(i, j - i));  // Extract the length of the next string
            i = j + 1;  // Move past the '#' character
            string str = s.substr(i, length);  // Extract the string of 'length' size
            res.push_back(str);
            i = i + length;  // Move to the next part
        }

        return res;
    }
};

int main() {
    Codec codec;

    // Test case
    vector<string> strs = {"neet", "code", "love", "you"};
    string encoded = codec.encode(strs);
    cout << "Encoded: " << encoded << endl;

    vector<string> decoded = codec.decode(encoded);
    cout << "Decoded: ";
    for (const string& s : decoded) {
        cout << s << " ";
    }
    cout << endl;

    return 0;
}


// g++ -std=c++17 Leetcode_0271_Encode_and_Decode_Strings.cpp -o test