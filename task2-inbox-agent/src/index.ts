import { z } from "zod";
import { mkdir, writeFile } from "node:fs/promises";

const EmailSchema = z.object({
  from: z.string().email(),
  subject: z.string().min(1),
  body: z.string().min(1),
});

type Email = z.infer<typeof EmailSchema>;

const knowledgeUrl =
  process.env.KNOWLEDGE_BRAIN_URL ?? "http://127.0.0.1:8000";

const variations = [
  "Could you please guide me?",
  "I would appreciate clear instructions.",
  "Please help me understand the steps.",
  "Could you explain how to do this?",
  "Please provide the relevant information.",
];

const scenarios = [
  {
    subject: "Import members using CSV",
    body: "How can I import members into the system using a CSV file?",
    supported: true,
  },
  {
    subject: "CSV member upload instructions",
    body: "Please explain how to upload a CSV file to import members.",
    supported: true,
  },
  {
    subject: "Trouble importing members",
    body: "I need help with the process for importing members via CSV.",
    supported: true,
  },
  {
    subject: "Member list import",
    body: "What is the process for importing a list of members from CSV?",
    supported: true,
  },
  {
    subject: "Bulk member import",
    body: "How do I add multiple members using a CSV import?",
    supported: true,
  },
  {
    subject: "Request for a refund",
    body: "I would like to request a refund for my recent payment.",
    supported: false,
  },
  {
    subject: "Website logo update",
    body: "How do I change the logo displayed on my website?",
    supported: false,
  },
  {
    subject: "Payroll question",
    body: "How should I calculate payroll taxes for my employees?",
    supported: false,
  },
  {
    subject: "Account security concern",
    body: "I think someone accessed my account without permission.",
    supported: false,
  },
  {
    subject: "Product pricing request",
    body: "Can you provide a price quote for a custom product?",
    supported: false,
  },
];

function classify(subject: string, body: string): string {
  const text = `${subject} ${body}`.toLowerCase();

  if (/refund|cancel|chargeback/.test(text)) return "refund_or_cancellation";
  if (/broken|error|not working|failed|trouble/.test(text)) return "troubleshooting";
  if (/price|quote|estimate|cost|pricing/.test(text)) return "quote_request";
  if (/complaint|unhappy|disappointed/.test(text)) return "complaint";
  if (/security|accessed my account|password|unauthorized/.test(text)) return "account_security";
  if (/how do i|how can i|where do i|please explain|what is the process/.test(text)) return "how_to";

  return "general";
}

async function main() {
  const emails: Array<Email & { expectedSupported: boolean }> = [];

  // Ten scenarios with five wording variations produce 50 sample emails.
  for (let scenarioIndex = 0; scenarioIndex < scenarios.length; scenarioIndex++) {
    const scenario = scenarios[scenarioIndex];

    for (let variantIndex = 0; variantIndex < variations.length; variantIndex++) {
      const email = EmailSchema.parse({
        from: `customer${String(scenarioIndex * 5 + variantIndex + 1).padStart(3, "0")}@example.com`,
        subject: scenario.subject,
        body: `${scenario.body} ${variations[variantIndex]}`,
      });

      emails.push({
        ...email,
        expectedSupported: scenario.supported,
      });
    }
  }

  const results = [];

  for (let index = 0; index < emails.length; index++) {
    const email = emails[index];
    console.log(`Processing email ${index + 1}/${emails.length}: ${email.subject}`);

    try {
      const response = await fetch(`${knowledgeUrl}/query`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question: `${email.subject}\n${email.body}`, top_k: 4 }),
        signal: AbortSignal.timeout(15000),
      });

      if (!response.ok) {
        throw new Error(`Knowledge API returned HTTP ${response.status}`);
      }

      const knowledge = await response.json() as {
        answer: string;
        status: string;
        sources: Array<{ title?: string; url?: string; relevance?: number }>;
        needs_human: boolean;
      };

      // This evaluation knowledge base supports CSV member-import questions only.
      // Route unrelated topics to human review.
      const csvImportQuestion =
        /\b(csv|spreadsheet|member list|member import|import members|bulk import|upload members)\b/i.test(
          `${email.subject}\n${email.body}`,
        );

      const hasRelevantSource = (knowledge.sources ?? []).some((source) =>
        /import members via csv|csv.*member|member.*csv/i.test(source.title ?? ""),
      );

      const answered =
        knowledge.status === "answered" &&
        !knowledge.needs_human &&
        csvImportQuestion &&
        hasRelevantSource;

      const actualStatus = answered ? "answered" : "insufficient_evidence";

      const draft = answered
        ? `Hello,\n\nThank you for contacting us. Here is the information we found:\n\n${knowledge.answer}\n\nPlease let us know if you need further assistance.`
        : "Hello,\n\nThank you for contacting us. A support specialist needs to review your question before we can provide an accurate answer.";

      results.push({
        emailNumber: index + 1,
        originalEmail: {
          from: email.from,
          subject: email.subject,
          body: email.body,
        },
        category: classify(email.subject, email.body),
        expectedSupported: email.expectedSupported,
        actualStatus,
        needsHumanExpert: !answered,
        needsHumanApproval: true,
        action: "DRAFT_ONLY - no email was sent",
        draft,
        sources: knowledge.sources,
        passed:
          email.expectedSupported
            ? answered
            : !answered,
      });
    } catch (error) {
      results.push({
        emailNumber: index + 1,
        originalEmail: {
          from: email.from,
          subject: email.subject,
          body: email.body,
        },
        category: classify(email.subject, email.body),
        expectedSupported: email.expectedSupported,
        actualStatus: "service_error",
        needsHumanExpert: true,
        needsHumanApproval: true,
        action: "DRAFT_ONLY - no email was sent",
        draft: "A support specialist must review this email because the knowledge service did not return a usable response.",
        sources: [],
        passed: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  const passed = results.filter((result) => result.passed).length;
  const summary = {
    totalEmails: results.length,
    passed,
    failed: results.length - passed,
    accuracyPercent: Number(((passed / results.length) * 100).toFixed(1)),
    supportedCases: results.filter((result) => result.expectedSupported).length,
    unsupportedCases: results.filter((result) => !result.expectedSupported).length,
    note: "Synthetic evaluation dataset. All responses are drafts; no emails are sent.",
  };

  await mkdir("output", { recursive: true });
  await writeFile(
    "output/batch_results.json",
    JSON.stringify({ summary, results }, null, 2),
    "utf-8",
  );

  console.log("\n--- Batch evaluation summary ---");
  console.log(JSON.stringify(summary, null, 2));
  console.log("\nDetailed report: output/batch_results.json");
}

main().catch((error) => {
  console.error("Batch processing failed:", error);
  process.exitCode = 1;
});
