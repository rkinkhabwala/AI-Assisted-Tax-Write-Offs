-- 0003: grounding-verification results per request (spec section 6; phase 8 metrics).

ALTER TABLE agent_requests
    ADD COLUMN verification_status   text
        CHECK (verification_status IN ('verified', 'revised', 'partially_verified', 'skipped', 'error')),
    ADD COLUMN verification_rounds   integer,
    ADD COLUMN claims_supported      integer,
    ADD COLUMN claims_partial        integer,
    ADD COLUMN claims_unsupported    integer,
    ADD COLUMN unknown_citations     text[],
    ADD COLUMN untraced_numbers      text[],
    ADD COLUMN verifier_input_tokens  integer,
    ADD COLUMN verifier_output_tokens integer;
