
这个算法可以快速计算对应的常系数齐次线性递推的第`n`项，复杂为$O(k^2logn)$，相比矩阵快速幂的$O(k^3logn)$还是有比较大的常数优势的，而且实际上也不是很难理解。

---

# 代码模板

```cpp
#include <bits/stdc++.h>
using namespace std;
using int64 = long long;

// Simple, efficient Kitamasa implementation for linear recurrences.
// Returns f(n) for recurrence of order k:
//   f(t) = c[0]*f(t-1) + c[1]*f(t-2) + ... + c[k-1]*f(t-k)
// given initial values f(0..k-1) and coefficients c[0..k-1].
// Complexity: O(k^2 * log n)

// Multiply two polynomials a and b (both size k) modulo the characteristic
// polynomial determined by c[], and reduce degree to < k.
static vector<int64> combine(const vector<int64>& a, const vector<int64>& b,
                             const vector<int64>& c, int64 mod) {
    int k = (int)c.size();
    vector<int64> tmp(2 * k, 0);
    for (int i = 0; i < k; ++i) if (a[i]) {
        for (int j = 0; j < k; ++j) if (b[j]) {
            __int128 v = (__int128)a[i] * b[j] + tmp[i + j];
            tmp[i + j] = (int64)(v % mod);
        }
    }
    // reduce terms with degree >= k using recurrence coefficients
    for (int d = 2 * k - 2; d >= k; --d) if (tmp[d]) {
        for (int j = 1; j <= k; ++j) {
            // x^d -> x^{d-j} * c[j-1]
            __int128 v = (__int128)tmp[d] * c[j - 1] + tmp[d - j];
            tmp[d - j] = (int64)(v % mod);
        }
    }
    vector<int64> res(k);
    for (int i = 0; i < k; ++i) res[i] = tmp[i] % mod;
    return res;
}

// kitamasa_pow(n) returns a vector coef of size k such that
// f(n) = sum_{i=0..k-1} coef[i] * f(i)
static vector<int64> kitamasa_pow(long long n, const vector<int64>& c, int64 mod) {
    int k = (int)c.size();
    if (n < k) {
        vector<int64> res(k, 0);
        res[n] = 1;
        return res;
    }
    if (n % 2 == 0) {
        auto half = kitamasa_pow(n / 2, c, mod);
        return combine(half, half, c, mod);
    } else {
        auto prev = kitamasa_pow(n - 1, c, mod);
        // multiply prev by x (i.e. shift). Represent poly 'x' as base.
        vector<int64> base(k, 0);
        if (k == 1) {
            // x mod (x - c0) == c0
            base[0] = c[0] % mod;
        } else {
            base[1] = 1;
        }
        return combine(prev, base, c, mod);
    }
}

// Main helper: compute f(n)
int64 kitamasa(const vector<int64>& init, const vector<int64>& coeffs, long long n, int64 mod = 1000000007LL) {
    int k = (int)coeffs.size();
    if (n < (int)init.size()) return (init[n] % mod + mod) % mod;
    auto coef = kitamasa_pow(n, coeffs, mod);
    __int128 ans = 0;
    for (int i = 0; i < k; ++i) {
        ans += (__int128)coef[i] * (init[i] % mod);
    }
    return (int64)(ans % mod);
}

// Example usage
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Example: Fibonacci
    // f(0)=0, f(1)=1, f(n)=f(n-1)+f(n-2)
    vector<int64> init = {0, 1};
    vector<int64> coeffs = {1, 1}; // f(n)=1*f(n-1)+1*f(n-2)
    long long n;
    cout << "Enter n to compute Fibonacci f(n) mod 1e9+7: ";
    if (!(cin >> n)) return 0;
    cout << kitamasa(init, coeffs, n) << '\n';
    return 0;
}
```

