
import argparse
import numpy as np

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter

class LMFilter(BaseFilter):
    name = "LM Filter"
    def __init__(
        self,
        exclusion_writer: DiskWriter = None,
        doc_length_threshold: int = 256,
        included_threshold: float = 0.5,
        random_seed: int = 42,
    ):
        """
        filters if the predicted language is not among given language or if the language score is below language
        language_threshold
        """
        super().__init__(exclusion_writer)
        self.included_threshold = included_threshold
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)
        self.doc_length_threshold = doc_length_threshold

    def filter(self, doc: Document) -> bool:
        """Args:
            doc: document

        Returns:
            is_filter
        """
        if len(doc.text) < self.doc_length_threshold:
            return False, "too_short_document"
        
        high_quality_prob = doc.metadata['fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob']
        if high_quality_prob > self.included_threshold:
            # if the quality score is above the included threshold, we include the document
            return True
        else:
            if high_quality_prob > 0.09532:
                normalized_prob = high_quality_prob ** 2
                # if the quality score is above the sampled threshold, we sample the document based on the score
                choice = self.random_state.choice([True, False], p=[normalized_prob, 1 - normalized_prob])
                if choice:
                    return True
                else:
                    return False, "score_below_threshold"
            else:
                return False, "score_below_threshold"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process FineWeb dataset")
    parser.add_argument("--num_tasks", type=int, default=32, help="Number of tasks")
    parser.add_argument("--num_local_tasks", type=int, default=-1, help="Number of local tasks")
    parser.add_argument("--local_rank_offset", type=int, default=0, help="Offset for the local rank")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers")
    parser.add_argument("--data_path", type=str, default="./data_raw", help="Path to the data")
    parser.add_argument("--save_path", type=str, default="./data_filtered", help="Path to the output")
    args = parser.parse_args()

    SAVE_DATA_PATH = f"{args.save_path}/data"
    REMOVED_DATA_PATH = f"{args.save_path}/removed"
    LOG_PATH = f"{args.save_path}/logs"

    main_processing_executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(args.data_path, compression="infer"),
            LMFilter(),
            JsonlWriter(SAVE_DATA_PATH, output_filename="filtered.chunk.${rank}.jsonl", compression="infer"),
        ],
        tasks=args.num_tasks,
        workers=args.num_workers,
        logging_dir=LOG_PATH,
        local_tasks=args.num_local_tasks,
        local_rank_offset=args.local_rank_offset,
        randomize_start_duration=0,
    )
    main_processing_executor.run()
