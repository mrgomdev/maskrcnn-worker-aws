# Worker for Mask R-CNN with AWS SQS and S3
1.  Fetch a request from the SQS
1.  Load image in the bucket with key /inputs/[request_id]
1.  Segment.
1.  Save image in the bucket with key /outputs/[request_id]
 
## Usage
```bash
AWS_PROFILE=[aws_credential_profile_name] A_BUCKET=[bucket_name_for_images] A_QUEUE=[sqs_name] python server.py
```

## (Want) to do
- [ ] Use tempfile
- [ ] Process on batch
- [ ] Tight sized output

## Reference
Model and inference part from [https://github.com/matterport/Mask_RCNN]