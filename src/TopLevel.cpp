/*
 * CUDARewriter.cpp
 *
 *  Created on: 24/aug/15
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

//TODO this is mandatory?
static llvm::cl::OptionCategory MatcherSampleCategory("Matcher Sample");
class MyASTConsumer : public ASTConsumer {
public:
	MyASTConsumer(CompilerInstance *comp, Rewriter &R) : ASTConsumer(), CI(comp)  { }
	virtual ~MyASTConsumer() { }

	virtual void Initialize(ASTContext &Context) {
		SM = &Context.getSourceManager();
		LO = &CI->getLangOpts();
		PP = &CI->getPreprocessor();
		//PP->addPPCallbacks(new RewriteIncludesCallback(this));
	}

	virtual bool HandleTopLevelDecl(DeclGroupRef DG) {

		Decl *firstDecl = DG.isSingleDecl() ? DG.getSingleDecl() : DG.getDeclGroup()[0];
		SourceLocation loc = firstDecl->getLocation();
		SourceLocation sloc = SM->getSpellingLoc(loc);

		std::cout << "loc " << loc.printToString(*SM) << "\n";
		std::cout << "sloc " << sloc.printToString(*SM) << "\n";

		//TODO bug when including header with the kernel definitions
		// (and probably all the attributed functions)
		// FIXable using cuda_runtime.h (seems like) but then another bug arises
		// because of some headers inclusions (happens only with our tools, not
		// using clang natively and dumping the asts)
		std::cout << SM->getFilename(loc).str() << "\n";

		if(SM->getFileID(loc) != SM->getMainFileID() && SM->getFileID(sloc) != SM->getMainFileID()){
			std::cout << "Skipping file " << loc.printToString(*SM) << "\n";
			return true;
		}
		std::cout << loc.printToString(*SM) << " --> Action required!\n";
		return true;
	}

private:
    CompilerInstance *CI;
    SourceManager *SM;
    LangOptions *LO;
    Preprocessor *PP;
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
