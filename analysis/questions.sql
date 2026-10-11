SELECT model,gsm8k,gsm1k,gsm1k-gsm8k accuracy_change,RANK() OVER(ORDER BY gsm8k DESC) original_rank,RANK() OVER(ORDER BY gsm1k DESC) replica_rank FROM observations ORDER BY original_rank,model;
SELECT COUNT(*) models,AVG(gsm1k-gsm8k) mean_change,MIN(gsm1k-gsm8k) worst_change FROM observations;
