import json
import argparse
import csv
import regex as re

def clean(s):
    res = re.sub(r'[^\w\s]', '', s)
    return res

def convert_ambig_to_tsv(inputFile, outputFile):
    with open(inputFile, 'r') as fp:
        data_dict = json.load(fp)
    print(len(data_dict))
    print(data_dict[0].keys())
    # used_queries = data_dict[0]['used_queries'][0]
    outputFp = open(outputFile, 'w')
    tsv_writer = csv.writer(outputFp, delimiter='\t')
    for ix in range(len(data_dict)):
        id = data_dict[ix]['id']
        query = data_dict[ix]['question']
        print(id, query)
        query = clean(query.strip())
        tsv_writer.writerow([ix, query])
    outputFp.close()
    return

def convert_webqsp_to_tsv(inputFile, outputFile):
    with open(inputFile, 'r') as fp:
        data_dict = json.load(fp)
    outputFp = open(outputFile, 'w')
    tsv_writer = csv.writer(outputFp, delimiter='\t')
    for ix in range(len(data_dict['Questions'])):
        query = data_dict['Questions'][ix]['ProcessedQuestion']
        tsv_writer.writerow([ix, query])
    outputFp.close()

def convert_ambig_data_to_json(inputFile, outputFile):
    with open(inputFile, 'r') as fp:
        data_dict = json.load(fp)
    outputDict = {}
    for ix in range(len(data_dict)):
        id = data_dict[ix]['id']
        query = data_dict[ix]['question']
        query = clean(query.strip())
        outputDict[ix] = {}
        outputDict[ix]['title'] = query
        outputDict[ix]['answers'] = data_dict[ix]['nq_answer']
    outputFp = open(outputFile, 'w')
    json.dump(outputDict, outputFp, indent = 6)
    outputFp.close()
    return

def convert_webqsp_data_to_json(inputFile, outputFile):
    with open(inputFile, 'r') as fp:
        data_dict = json.load(fp)
    outputDict = {}
    for ix in range(len(data_dict['Questions'])):
        query = data_dict['Questions'][ix]['ProcessedQuestion']
        parseList = data_dict['Questions'][ix]['Parses']
        ans_list = set()
        for parse in parseList:
            _ansList = parse['Answers']
            for ansdict in _ansList:
                ans_list.add(ansdict['EntityName'])
        outputDict[ix] = {}
        outputDict[ix]['title'] = query
        outputDict[ix]['answers'] = list(ans_list)
    outputFp = open(outputFile, 'w')
    json.dump(outputDict, outputFp, indent = 6)
    outputFp.close()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required = False, default = '../data/queries/ambig-dev.json', help = 'Input json file')
    parser.add_argument('--type', required = False, default = "ambig", help = 'type of dataset')
    parser.add_argument('--output', required = False, default = '../data/queries/ambig-dev.tsv', help = 'Output tsv file')
    args = parser.parse_args()
    if args.type == "ambig":
        convert_ambig_to_tsv(args.input, args.output)
    elif args.type == "webqsp":
        convert_webqsp_to_tsv(args.input, args.output)
    elif args.type == "trec-ambig":
        convert_ambig_data_to_json(args.input, args.output)
    elif args.type == "trec-webqsp":
        convert_webqsp_data_to_json(args.input, args.output)
