import json
import sys
sys.path += ['../']
from utils.dpr_utils import  get_model_obj, SimpleTokenizer, has_answer



def validate(pred_file_path, n_docs, dpr_result=False):
    print ('Validating: ', pred_file_path, flush=True)
    prediction = json.load(open(pred_file_path))

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)



    print('Matching answers in top docs...')
    scores = []
    for query_ans_doc in prediction:
        question=query_ans_doc['question']
        answers=query_ans_doc['answers']
        if not dpr_result:
            cxts=query_ans_doc['ctxs'][:n_docs]
        else:
            try:
                cxts = [d for d in query_ans_doc['positive_ctxs'] if d['title_score'] != 1.0] + query_ans_doc['negative_ctxs']
                cxts.sort(reverse=True, key=lambda d: d['score'])
                cxts = cxts[:n_docs]
            except:
                cxts=query_ans_doc['ctxs'][:n_docs]


        hits = []

        for i, doc in enumerate(cxts):
            if dpr_result: doc=doc['text']
            hits.append(has_answer(answers, doc, tokenizer))
        scores.append(hits)

    print('Per question validation results len=%d', len(scores))


    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    print('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v/len(prediction) for v in top_k_hits]
    print('Validation results: top k documents hits accuracy %s', top_k_hits)
    print('Validation results: top 20/100 documents hits accuracy ', top_k_hits[19], '/', top_k_hits[99])

    return top_k_hits






if __name__ == '__main__':


    # pred_file_path='/checkpoint/mdrizwanparvez/data/first_round_dpr/NQ/DPR_REPROD.dpr_reproduction_test.nq.bert-base-uncased.output.nopp.title.onlypassages.json.converted.json'
    # pred_file_path='/checkpoint/mdrizwanparvez/data/first_round_dpr/NQ/'
    # pred_file_path='/checkpoint/mdrizwanparvez/data/first_round_dpr/NQ/AFTER_CHAT.ANCE_dev_singleset.nq.bert-base-uncased.output.nopp.title.onlypassages.json.converted.json'
    # validate(pred_file_path.strip(), 100, dpr_result=True)


    pred_file_path='/checkpoint/sviyer/wiki_psg100/ttttt_test.nq.bbase.output.nopp.title.onlypassages.ctxloss.scottformat.json'
    pred_file_path='/checkpoint/mdrizwanparvez/data/first_round_dpr/NQ/ANSE_SINGLESET_RERANKER_TRAINED_ON_ANCE.dev.nq.bert-base-uncased.output.nopp.title.onlypassages.json.converted.json'
    pred_file_path='/checkpoint/mdrizwanparvez/data/first_round_dpr/NQ/FINAL_ANCE_RERANKER.test.nq.bert-base-uncased.output.nopp.title.onlypassages.json.converted.json'
    pred_file_path='/checkpoint/mdrizwanparvez/data/first_round_dpr/NQ/FINAL_ANCE_SINGLESET_DEV..bug_fixed.dev.nq.bert-base-uncased.output.nopp.title.onlypassages.json.converted.json'


    # pred_file_path='/checkpoint/mdrizwanparvez/data/first_round_dpr/NQ/Correct.BUG_FIXED_CLUSTERING..dev.nq.bert-base-uncased.output.nopp.title.onlypassages.json.dprformat.json'
    # validate(pred_file_path.strip(), 100, dpr_result=True)
    pred_file_path='/checkpoint/mdrizwanparvez/data/first_round_dpr/NQ/Correct.BUG_FIXED_CLUSTERING..test.nq.bert-base-uncased.output.nopp.title.onlypassages.json.dprformat.json'
    validate(pred_file_path.strip(), 100, dpr_result=True)
    # #
    # pred_file_path='/private/home/mdrizwanparvez/ANCE/commands/NQ_prediction_ouput_file.json'
    # validate(pred_file_path, 100)






    # pred_file_path='/checkpoint/mdrizwanparvez/data/first_round_dpr/NQ/ctx_to_q_loss_test.nq.bert-base-uncased.output.nopp.title.onlypassages.json.converted.json'
    # validate(pred_file_path, 100, dpr_result=True)

    # pred_file_path='/checkpoint/sviyer/wiki_psg100/ttttt_dev.nq.bbase.output.nopp.title.onlypassages.ctxloss.scottformat.json'
    # validate(pred_file_path, 100, dpr_result=True)
    # pred_file_path='/checkpoint/sviyer/wiki_psg100/ttttt_test.nq.bbase.output.nopp.title.onlypassages.ctxloss.scottformat.json'
    # validate(pred_file_path, 100, dpr_result=True)



    # pred_file_path = '/private/home/mdrizwanparvez/ANCE/commands/TRIVIAQ_prediction_ouput_file.json'
    # validate(pred_file_path, 100 )
    
    #DPR rernaker 
    dset = 'nq'
    train='/checkpoint/mdrizwanparvez/data/first_round_dpr/NQ/train_reformat.json'
    dev='/checkpoint/mdrizwanparvez/data/first_round_dpr/NQ/dev_reformat.json'

    # python train_qa.py --prefix wiki.dpr.67k.qpadding. --tfile $train  --efile $dev  --extra ' --pad_question --max_question_length 20 --max_passage_length 220  ' --partition dev
    model='/checkpoint/midrizwanparvez/model/models_qa_wiki.dpr.67k.qpadding._nq_tbz_16_ebz_144_m_20_g8__--pad_question_--max_question_length_20_--max_pas_/best-model-27000.pt'

    bert = 'bert-base-uncased'
    partition = 'dev'


    # pred_path ='/checkpoint/mdrizwanparvez/data/first_round_dpr/NQ/ttttt_dev.'+dset+'.'+bert+'.output.nopp.title.onlypassages.json'
    # converted_pred_path='/checkpoint/mdrizwanparvez/data/first_round_dpr/NQ/ttttt_dev.'+dset+'.'+bert+'.output.nopp.title.onlypassages.converted.json'
    # validate(converted_pred_path, 100, dpr_result=True)
    # dpr_dev_path = '/checkpoint/mdrizwanparvez/hf_bert/validate_nq_dev/nq-dev-dense-results.json'
    # validate(dpr_dev_path, 100, dpr_result=True)

    # test_dev_path = '/checkpoint/mdrizwanparvez/hf_bert/nq-test-dense-results.json'
    # # validate(test_dev_path, 100, dpr_result=True)



    # converted_pred_path='/checkpoint/sviyer/wiki_psg100/ttttt_dev.nq..output.nopp.title.onlypassages.converted.json'
    # validate(converted_pred_path, 100, dpr_result=True)

    dpr_dev_path = '/private/home/mdrizwanparvez/ODT/data/retriever_results/nq/single/dev.json'
    # validate(dpr_dev_path, 100, dpr_result=True)

    ANCE_TEST_PATh='/checkpoint/mdrizwanparvez/data/first_round_dpr/NQ/ANCE_test.nq.bert-base-uncased.output.nopp.title.onlypassages.converted.json'
    # validate(ANCE_TEST_PATh, 100, dpr_result=True)





    