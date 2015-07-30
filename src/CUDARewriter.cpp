/*
 * CUDARewriter.cpp
 *
 *  Created on: 24/lug/2015
 *      Author: latzori
 */


#include <string>
#include <iostream>
#include <list>
#include <map>
#include <set>
#include <sstream>

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "llvm/Support/raw_ostream.h"


#include "clang/AST/Decl.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PPCallbacks.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Regex.h"

using namespace clang;
using namespace clang::ast_matchers;
using namespace clang::driver;
using namespace clang::tooling;
using namespace llvm::sys::path;

static llvm::cl::OptionCategory MatcherSampleCategory("Matcher Sample");

class KernelDeclHandler : public MatchFinder::MatchCallback {
public:
	KernelDeclHandler(Rewriter &Rewrite, Preprocessor* PP) : Rewrite(Rewrite), PP(PP){}

	virtual void run(const MatchFinder::MatchResult &Result){
		clang::SourceManager* const SM = Result.SourceManager;
		if (const clang::FunctionDecl * kerneldecl = Result.Nodes.getNodeAs<clang::FunctionDecl>("CUDA_kernel_functionDecl")){
			std::string SStr;
			llvm::raw_string_ostream S(SStr);
			kerneldecl->print(S);
			std::cout << "kerneldecl is " << S.str() << "\n";
			if(CUDAGlobalAttr* gattr = kerneldecl->getAttr<CUDAGlobalAttr>()){
				SourceLocation gattrloc = SM->getExpansionLoc(gattr->getLocation());
				if(SM->getFileID(gattrloc) != SM->getMainFileID()){printf("gattrloc: Different IDs\n");} else{
				SourceLocation gattrend = PP->getLocForEndOfToken(gattrloc);
				SourceRange sr(gattrloc, gattrend);
				Rewrite.ReplaceText(gattrloc, Rewrite.getRangeSize(sr),	"");
				}
			}
		}
	}
private:
	Rewriter &Rewrite;
	Preprocessor* PP;
};

class AttributeHandler : public MatchFinder::MatchCallback {
public:
	AttributeHandler(Rewriter &Rewrite, Preprocessor* PP) : Rewrite(Rewrite), PP(PP){}

	virtual void run(const MatchFinder::MatchResult &Result){
		clang::SourceManager* const SM = Result.SourceManager;

		//Deleting __host__ attribute
		if(const clang::FunctionDecl * hostfunc = Result.Nodes.getNodeAs<clang::FunctionDecl>("CUDAHost_Attr_functionDecl")){

			std::string SStr;
			llvm::raw_string_ostream S(SStr);
			hostfunc->print(S);
			std::cout << "hostfunc is " << S.str() << "\n";
			if(CUDAHostAttr* hattr = hostfunc->getAttr<CUDAHostAttr>()){
				SourceLocation hattrloc = SM->getExpansionLoc(hattr->getLocation());
				if(SM->getFileID(hattrloc) != SM->getMainFileID()){printf("hattrloc: Different IDs\n");} else{
				SourceLocation hattrend = PP->getLocForEndOfToken(hattrloc);
				SourceRange sr(hattrloc, hattrend);
				Rewrite.ReplaceText(hattrloc, Rewrite.getRangeSize(sr),	"");
				}
			}
		}

		//Deleting __device__ attribute
		if (const clang::FunctionDecl * devicefunc = Result.Nodes.getNodeAs<clang::FunctionDecl>("CUDADevice_Attr_functionDecl")){
			std::string SStr2;
			llvm::raw_string_ostream S2(SStr2);
			devicefunc->print(S2);
			std::cout << "devicefunc is " << S2.str() << "\n";

			if(CUDADeviceAttr* dattr = devicefunc->getAttr<CUDADeviceAttr>()){
				SourceLocation dattrloc = SM->getExpansionLoc(dattr->getLocation());

				if(SM->getFileID(dattrloc) != SM->getMainFileID()){printf("dattrloc: Different IDs\n");} else{

				SourceLocation dattrend = PP->getLocForEndOfToken(dattrloc);
				SourceRange sr(dattrloc, dattrend);
				Rewrite.ReplaceText(dattrloc, Rewrite.getRangeSize(sr),	"");}
			}
		}

		//And now we have to translate the body of the function!

	}

private:
	Rewriter &Rewrite;
	Preprocessor *PP;
};
/**
*
* Our attempt is to write only matchers for entry points, and then
* recursively call more specific functions from the handlers!
*
* TODO list:
* - definizioni di kernel (dopo traduco il body)
* - definizioni di funzioni con attributi __device__ o __host__ (rimuovo
* 	l'attributo e poi traduco il body)
* - il main come lo tratto? perché dentro avrò sintassi cuda...
*
*/
class MyASTConsumer : public ASTConsumer {
public:
	MyASTConsumer(CompilerInstance *comp, Rewriter &R) : CI(comp)  {

		//Preprocessor
		Preprocessor* P = &CI->getPreprocessor();

		//Initialize all the handlers!
		AH = new AttributeHandler(R, P);
		KDH = new KernelDeclHandler(R, P);

		//AH2 = new AttributeHandler(R, P);

		//Matches all the function declarations with an __host__ attribute
		Matcher.addMatcher(
				functionDecl(
						hasAttr(
								clang::attr::CUDAHost
								)
						).bind("CUDAHost_Attr_functionDecl"),
				AH);

		//Matches all the function declarations with a __device__ attribute
		Matcher.addMatcher(
				functionDecl(
						hasAttr(
								clang::attr::CUDADevice
								)
						).bind("CUDADevice_Attr_functionDecl"),
				AH);

		//Matches all the function declarations with a __global__ attribute (kernels?)
		Matcher.addMatcher(
				functionDecl(
						hasAttr(
								clang::attr::CUDAGlobal
								)
						).bind("CUDA_kernel_functionDecl"),
				KDH);
		printf("ASTConsumer: added matchers\n");

	}

	// Run the matchers when we have the whole TU parsed.
	void HandleTranslationUnit(ASTContext &Context) override {
		Matcher.matchAST(Context);

	}

private:
	MatchFinder Matcher;

	AttributeHandler *AH;

	KernelDeclHandler *KDH;

	//Preprocessor* PP;
	CompilerInstance *CI;

};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {}
  void EndSourceFileAction() override {
    TheRewriter.getEditBuffer(TheRewriter.getSourceMgr().getMainFileID())
        .write(llvm::outs());
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef file) override {
    TheRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return llvm::make_unique<MyASTConsumer>(&CI, TheRewriter);
  }

private:
  Rewriter TheRewriter;
};

int main(int argc, const char **argv) {
  CommonOptionsParser op(argc, argv, MatcherSampleCategory);
  ClangTool Tool(op.getCompilations(), op.getSourcePathList());

  return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
