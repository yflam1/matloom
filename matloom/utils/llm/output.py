from datetime import datetime, timezone
from typing import Annotated, Any, Generic, Literal, TypeVar
from uuid import uuid4

from openai.types import ReasoningEffort
from openai.types.chat import ChatCompletionTokenLogprob
from pydantic import BaseModel, Field, HttpUrl

from ..dtypes import NonEmptyStr, SDict
from .msg import Messages
from .response import ResponseModel

# A response is either a parsed ``ResponseModel`` or the raw assistant string.
ResponseType = TypeVar("ResponseType", bound=ResponseModel | str)


class LlmChatError(RuntimeError):
    """Raised for *semantic* failures of a completion (empty/None response,
    missing logprobs, …). These are retried locally; transport errors
    (rate-limit / 5xx / connection) are left to the OpenAI SDK's own retry
    machinery instead of being retried a second time here."""


class ChatParameters(BaseModel, validate_assignment=True, strict=True):
    model: NonEmptyStr
    max_completion_tokens: Annotated[int, Field(ge=1)] | None = None
    timeout: Annotated[float, Field(gt=0.0)] | None = None
    max_retries: Annotated[int, Field(ge=0)] | None = None
    base_url: HttpUrl | None = None
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] | None = None
    image_detail: Literal["auto", "low", "high"] | None = None
    reasoning_effort: ReasoningEffort = None
    logprobs: bool | None = None
    top_logprobs: Annotated[int, Field(ge=0, le=5)] | None = None
    extra_body: SDict[Any] | None = None

    def to_api_params(self) -> SDict[Any]:
        # ``timeout``/``max_retries``/``base_url`` configure the client, not the
        # request; ``image_detail`` is applied when building the messages.
        return self.model_dump(
            exclude={"timeout", "max_retries", "base_url", "image_detail"},
            exclude_none=True,
        )


class LlmOutput(
    BaseModel, Generic[ResponseType], validate_assignment=True, strict=True
):
    id: NonEmptyStr = Field(default_factory=lambda: uuid4().hex, frozen=True)
    created_at: NonEmptyStr = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            r"%Y/%m/%d %H:%M:%S"
        ),
        frozen=True,
    )
    parameters: ChatParameters
    response: ResponseType | None = None
    content: NonEmptyStr | None = None
    detail: NonEmptyStr | None = None
    duration: Annotated[float, Field(gt=0.0)] | None = None
    finish_reason: NonEmptyStr | None = None
    prompt_tokens: Annotated[int, Field(ge=0)] | None = None
    completion_tokens: Annotated[int, Field(ge=0)] | None = None
    cost: Annotated[float, Field(ge=0.0)] | None = None
    logprobs: list[ChatCompletionTokenLogprob] | None = None
    messages: Messages | None = None
