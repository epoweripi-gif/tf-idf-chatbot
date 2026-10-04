#include <bits/stdc++.h>
using namespace std;

#ifdef _WIN32
    #define NOMINMAX
    #include <windows.h>
#endif

#ifdef _WIN32
    #define POPEN _popen
    #define PCLOSE _pclose
#else
    #define POPEN popen
    #define PCLOSE pclose
#endif

// ---------- put the model name from Groq's models page here ----------
const string MODEL_NAME = "openai/gpt-oss-20b";
const string API_URL = "https://api.groq.com/openai/v1/chat/completions";
// ---------------------------------------------------------------------

// ---------- typing effect ----------
void waitMs(int ms) {
    clock_t start = clock();
    while ((clock() - start) * 1000 / CLOCKS_PER_SEC < ms) {}
}

void typeOut(const string &text) {
    for (char c : text) { cout << c << flush; waitMs(20); }
    cout << endl;
}

// ---------- TF-IDF part ----------
vector<string> tokenize(const string &s) {
    string t;
    for (char c : s) t += isalnum((unsigned char)c) ? tolower(c) : ' ';
    stringstream ss(t);
    vector<string> words;
    string w;
    while (ss >> w) words.push_back(w);
    return words;
}

struct TfIdfBot {
    vector<string> answers;
    vector<map<string, double>> docVec;
    map<string, double> idf;
    double unknownIdf = 1.0;

    map<string, double> makeVec(const vector<string> &words) {
        map<string, double> v;
        for (auto &w : words) v[w] += 1.0 / words.size();
        // words the bot has never seen still count, so they lower the match
        for (auto &p : v)
            p.second *= idf.count(p.first) ? idf[p.first] : unknownIdf;
        return v;
    }

    double cosine(map<string, double> &a, map<string, double> &b) {
        double dot = 0, na = 0, nb = 0;
        for (auto &p : a) {
            na += p.second * p.second;
            if (b.count(p.first)) dot += p.second * b[p.first];
        }
        for (auto &p : b) nb += p.second * p.second;
        if (na == 0 || nb == 0) return 0;
        return dot / (sqrt(na) * sqrt(nb));
    }

    void build(const vector<pair<string, string>> &facts) {
        vector<vector<string>> docs;
        map<string, int> df;
        for (auto &f : facts) {
            docs.push_back(tokenize(f.first));
            answers.push_back(f.second);
            set<string> unique(docs.back().begin(), docs.back().end());
            for (auto &w : unique) df[w]++;
        }
        int N = facts.size();
        for (auto &p : df)
            idf[p.first] = log((N + 1.0) / (p.second + 1.0)) + 1.0;
        unknownIdf = log(N + 1.0) + 1.0;
        for (auto &d : docs) docVec.push_back(makeVec(d));
    }

    bool match(const string &message, string &answer) {
        map<string, double> q = makeVec(tokenize(message));
        double best = 0;
        int bestIndex = -1;
        for (int i = 0; i < (int)docVec.size(); i++) {
            double s = cosine(q, docVec[i]);
            if (s > best) { best = s; bestIndex = i; }
        }
        if (bestIndex == -1 || best < 0.3) return false;
        answer = answers[bestIndex];
        return true;
    }
};

// ---------- API part (no extra library) ----------

// make a text safe to put inside quotes in JSON
string jsonEscape(const string &s) {
    string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "";
        else if (c == '\t') out += "\\t";
        else if ((unsigned char)c < 32) out += ' ';
        else out += c;
    }
    return out;
}

// find the answer text inside Groq's reply
string readContent(const string &reply) {
    size_t pos = reply.find("\"content\"");
    if (pos == string::npos) return "";
    pos = reply.find(':', pos);
    if (pos == string::npos) return "";
    pos++;
    while (pos < reply.size() && reply[pos] == ' ') pos++;
    if (pos >= reply.size() || reply[pos] != '"') return "";
    pos++;

    string out;
    while (pos < reply.size() && reply[pos] != '"') {
        if (reply[pos] == '\\' && pos + 1 < reply.size()) {
            pos++;
            char c = reply[pos];
            if (c == 'n') out += '\n';
            else if (c == 't') out += '\t';
            else if (c == 'u' && pos + 4 < reply.size()) {
                int code = stoi(reply.substr(pos + 1, 4), nullptr, 16);
                out += (code < 128) ? (char)code : '?';
                pos += 4;
            }
            else out += c;            // handles \" and \\ and \/
        } else {
            out += reply[pos];
        }
        pos++;
    }
    return out;
}

vector<pair<string, string>> chatHistory;   // {role, text}

string askAI(const string &question) {
    const char *key = getenv("API_KEY");
    if (!key) { cout << "[debug] API_KEY is not set in this terminal." << endl; return ""; }
    // build the message by hand
    string body = "{\"model\":\"" + MODEL_NAME + "\",\"messages\":["
        "{\"role\":\"system\",\"content\":\"You are Simple Chatbot, a friendly helper built in C++. Keep answers short and simple.\"}";
    for (auto &m : chatHistory)
        body += ",{\"role\":\"" + m.first + "\",\"content\":\"" + jsonEscape(m.second) + "\"}";
    body += ",{\"role\":\"user\",\"content\":\"" + jsonEscape(question) + "\"}]}";

    ofstream("req.json") << body;

    string cmd = "curl -s \"" + API_URL + "\" "
        + "-H \"Content-Type: application/json\" "
        + "-H \"Authorization: Bearer " + key + "\" "
        + "-d @req.json";

    FILE *p = POPEN(cmd.c_str(), "r");
    if (!p) return "";
    string out;
    char buf[256];
    while (fgets(buf, sizeof buf, p)) out += buf;
    PCLOSE(p);

    string answer = readContent(out);
    if (answer.empty()) cout << "[debug] Groq said: " << out.substr(0, 300) << endl;
    return answer;
}

void remember(const string &question, const string &answer) {
    chatHistory.push_back({"user", question});
    chatHistory.push_back({"assistant", answer});
    while (chatHistory.size() > 6) chatHistory.erase(chatHistory.begin());
}

int main() {
    #ifdef _WIN32
    SetConsoleOutputCP(65001);   // lets the terminal show special characters
    #endif

    TfIdfBot bot;
    bot.build({
        {"hello hi hey greetings good morning", "Hi! Nice to meet you."},
        {"what is your name who are you", "I am Simple Chatbot, built in C++."},
        {"who made built created you", "I was built from scratch in C++."},
        {"what can you do help", "I can chat and answer a few questions."},
        {"how are you feeling today", "I am doing great, thanks for asking!"}
    });

    typeOut("HELLO THERE! I AM SIMPLE CHATBOT. TYPE BYE TO STOP.");

    string user;
    while (true) {
        cout << "You: ";
        getline(cin, user);

        string lower = user;
        transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower == "bye") break;

        string answer;
        if (!bot.match(user, answer)) {
            answer = askAI(user);
            if (answer.empty())
                answer = "Sorry, I could not get an answer. Check your key, model name and internet.";
        }
        remember(user, answer);

        cout << "Bot: ";
        typeOut(answer);
    }
    return 0;
}