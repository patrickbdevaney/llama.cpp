#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// check if the llama_context is compatible for speculative decoding
// note: clears the memory of the context
bool common_speculative_is_compat(llama_context * ctx_tgt);

// seq_id_tgt is the sequence this speculator drives on the target context. It only matters for
// targets whose memory cannot drop rejected tokens by itself (recurrent and hybrid models such as
// GLM-5.3): those are rolled back by checkpointing and replaying the recurrent state of that one
// sequence. Callers that run a single sequence can leave it at 0.
common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt,
        llama_seq_id                seq_id_tgt = 0);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// Tell a hidden-state speculator (MTP) which target output row holds the hidden state for the
// token preceding id_last - i.e. the number of draft tokens the target just accepted. Harmless
// and ignored for every other speculative type. Call it after sampling the target, before the
// next common_speculative_draft().
void common_speculative_set_target_output_idx(common_speculative * spec, int32_t i);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

// Finish the rollback that common_speculative_set_target_output_idx() started, on targets that
// need one (recurrent and hybrid models - see common_spec_rollback in speculative.cpp). The
// rejected tokens are already gone from the target's memory by then; what is left is a decode to
// replay the accepted ones, and a decode overwrites the context's output buffer. So the caller
// has to say when it is done reading logits for every sequence sharing that context.
//
// Call it once after the whole sample-and-accept pass. It is a no-op for ordinary targets and for
// sequences with nothing pending, and common_speculative_draft() calls it too, so forgetting it
// costs correctness only if the sequence is dropped before it ever drafts again.
void common_speculative_flush_rollback(common_speculative * spec);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);
