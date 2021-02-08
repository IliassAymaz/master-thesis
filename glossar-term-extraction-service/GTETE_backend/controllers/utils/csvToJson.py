import json

input_ = open('OpenReqs.csv', 'r').readlines()
with open('OpenReqs.json', 'w') as output:
    # list_ = [{key: value for key, value in zip(range(1, len(input_) + 1), input_)}]
    list_ = [
        {
            "id": str(id_),
            "text": text
        }
        for id_, text in zip(range(1, len(input_)+1), input_)
    ]
    json.dump(list_, output, indent = 2, ensure_ascii=False)
